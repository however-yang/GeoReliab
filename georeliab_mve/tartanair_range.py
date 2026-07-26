'''Validated HTTP-Range extraction of selected TartanAir ZIP members.

This is intentionally generic: an operator supplies the official archive URL,
the central directory is validated first, and only selected members are fetched.
It prevents a 20--35 GB whole-modality acquisition for the frozen P000 sanity
sample while retaining ZIP CRC, size, and ETag evidence.
'''

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Any, Callable, Sequence
import urllib.request
import zlib

from .preparation import PreparationError


@dataclass(frozen=True)
class RemoteZipEntry:
    name: str
    compression: int
    compressed_size: int
    uncompressed_size: int
    crc32: int
    local_offset: int


@dataclass(frozen=True)
class RemoteZipIndex:
    content_length: int
    etag: str | None
    entries: dict[str, RemoteZipEntry]
    central_directory_sha256: str


def _request_range(url: str, start: int, end: int) -> tuple[bytes, str | None, int]:
    if start < 0 or end < start:
        raise PreparationError('invalid HTTP byte range')
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(request) as response:
        if response.status != 206:
            raise PreparationError('official archive server must honor HTTP Range requests')
        content_range = response.headers.get('Content-Range', '')
        if '/' not in content_range:
            raise PreparationError('HTTP Range response omitted Content-Range total')
        try:
            total = int(content_range.rsplit('/', 1)[1])
        except ValueError as exc:
            raise PreparationError('HTTP Range response has invalid total length') from exc
        data = response.read()
        expected = end - start + 1
        if len(data) != expected:
            raise PreparationError(f'HTTP Range response length mismatch: {len(data)} != {expected}')
        return data, response.headers.get('ETag'), total


def _zip64_values(extra: bytes, needs: int) -> tuple[int, ...]:
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, size = struct.unpack_from('<HH', extra, cursor)
        value = extra[cursor + 4:cursor + 4 + size]
        cursor += 4 + size
        if kind == 0x0001:
            if len(value) < needs * 8:
                raise PreparationError('ZIP64 extra field is incomplete')
            return struct.unpack_from('<' + 'Q' * needs, value)
    raise PreparationError('ZIP64 archive is missing required extended offsets')


def _central_directory_location(url: str) -> tuple[int, int, int, str | None]:
    # EOCD is within the final 65,557 bytes unless archive comments are invalid.
    head = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(head) as response:
        length_header = response.headers.get('Content-Length')
        if not length_header:
            raise PreparationError('official archive did not provide Content-Length')
        total = int(length_header)
    tail_start = max(0, total - 65_557)
    tail, etag, checked_total = _request_range(url, tail_start, total - 1)
    if checked_total != total:
        raise PreparationError('archive Content-Length changed during central-directory read')
    marker = tail.rfind(b'PK\x05\x06')
    if marker < 0 or marker + 22 > len(tail):
        raise PreparationError('archive has no valid ZIP end-of-central-directory record')
    fields = struct.unpack_from('<4s4H2LH', tail, marker)
    directory_size, directory_offset = fields[5], fields[6]
    if directory_size != 0xFFFFFFFF and directory_offset != 0xFFFFFFFF:
        return directory_offset, directory_size, total, etag
    locator_start = tail_start + marker - 20
    locator, _, _ = _request_range(url, locator_start, locator_start + 19)
    if locator[:4] != b'PK\x06\x07':
        raise PreparationError('ZIP64 archive is missing end locator')
    zip64_offset = struct.unpack_from('<Q', locator, 8)[0]
    record, _, _ = _request_range(url, zip64_offset, zip64_offset + 55)
    if record[:4] != b'PK\x06\x06':
        raise PreparationError('ZIP64 archive has invalid end record')
    directory_size = struct.unpack_from('<Q', record, 40)[0]
    directory_offset = struct.unpack_from('<Q', record, 48)[0]
    return directory_offset, directory_size, total, etag


def index_remote_zip(url: str) -> RemoteZipIndex:
    '''Read and validate the official ZIP central directory without downloading it all.'''

    offset, size, total, etag = _central_directory_location(url)
    if size <= 0 or offset + size > total:
        raise PreparationError('central-directory range is outside official archive')
    raw, range_etag, checked_total = _request_range(url, offset, offset + size - 1)
    if checked_total != total or (etag and range_etag and etag != range_etag):
        raise PreparationError('official archive changed while reading central directory')
    entries: dict[str, RemoteZipEntry] = {}
    cursor = 0
    while cursor < len(raw):
        if raw[cursor:cursor + 4] != b'PK\x01\x02' or cursor + 46 > len(raw):
            raise PreparationError('central directory has invalid member header')
        header = struct.unpack_from('<4s6H3L5H2L', raw, cursor)
        flags, compression, crc32, compressed, uncompressed = header[3], header[4], header[7], header[8], header[9]
        name_len, extra_len, comment_len, local_offset = header[10], header[11], header[12], header[16]
        end = cursor + 46 + name_len + extra_len + comment_len
        if end > len(raw) or flags & 0x1:
            raise PreparationError('central directory has truncated or encrypted member')
        name = raw[cursor + 46:cursor + 46 + name_len].decode('utf-8')
        extra = raw[cursor + 46 + name_len:cursor + 46 + name_len + extra_len]
        missing = sum(value == 0xFFFFFFFF for value in (uncompressed, compressed, local_offset))
        if missing:
            values = iter(_zip64_values(extra, missing))
            if uncompressed == 0xFFFFFFFF:
                uncompressed = next(values)
            if compressed == 0xFFFFFFFF:
                compressed = next(values)
            if local_offset == 0xFFFFFFFF:
                local_offset = next(values)
        if name in entries:
            raise PreparationError(f'central directory has duplicate member: {name}')
        entries[name] = RemoteZipEntry(name, compression, compressed, uncompressed, crc32, local_offset)
        cursor = end
    return RemoteZipIndex(total, etag or range_etag, entries, hashlib.sha256(raw).hexdigest())


def _read_and_verify_existing(path: Path, entry: RemoteZipEntry) -> str:
    if path.stat().st_size != entry.uncompressed_size:
        raise PreparationError(f'existing selected member has wrong size: {entry.name}')
    digest = hashlib.sha256()
    crc = 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
            crc = zlib.crc32(block, crc)
    if crc & 0xffffffff != entry.crc32:
        raise PreparationError(f'existing selected member has wrong CRC: {entry.name}')
    return digest.hexdigest()


def extract_range_members_evidence(
    url: str, index: RemoteZipIndex, members: Sequence[str], destination: Path,
) -> dict[str, dict[str, Any]]:
    '''Fetch selected members atomically and retain ZIP/source evidence.'''

    evidence: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for name in members:
        if name in seen:
            raise PreparationError(f'duplicate selected ZIP member request: {name}')
        seen.add(name)
        entry = index.entries.get(name)
        if entry is None or name.startswith('/') or '..' in Path(name).parts:
            raise PreparationError(f'requested range member is not a safe indexed ZIP member: {name}')
        output = destination / name
        if output.exists():
            evidence[name] = _member_evidence(
                entry, output, _read_and_verify_existing(output, entry), 'reused'
            )
            continue
        header, etag, total = _request_range(url, entry.local_offset, entry.local_offset + 29)
        if total != index.content_length or (index.etag and etag and index.etag != etag) or header[:4] != b'PK\x03\x04':
            raise PreparationError('archive changed or local ZIP header is invalid')
        name_len, extra_len = struct.unpack_from('<HH', header, 26)
        local_name_raw, local_etag, local_total = _request_range(
            url, entry.local_offset + 30, entry.local_offset + 29 + name_len,
        )
        if local_total != index.content_length or (index.etag and local_etag and index.etag != local_etag):
            raise PreparationError('archive changed while verifying local ZIP member name')
        try:
            local_name = local_name_raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise PreparationError('local ZIP member name is not UTF-8') from exc
        if local_name != name:
            raise PreparationError(f'central/local ZIP member name mismatch: {name} != {local_name}')
        data_start = entry.local_offset + 30 + name_len + extra_len
        if entry.compressed_size:
            packed, etag, total = _request_range(url, data_start, data_start + entry.compressed_size - 1)
            if total != index.content_length or (index.etag and etag and index.etag != etag):
                raise PreparationError('archive changed during range member extraction')
        else:
            packed = b''
        if entry.compression == 0:
            data = packed
        elif entry.compression == 8:
            data = zlib.decompress(packed, -zlib.MAX_WBITS)
        else:
            raise PreparationError(f'unsupported ZIP compression for {name}: {entry.compression}')
        if len(data) != entry.uncompressed_size or zlib.crc32(data) & 0xffffffff != entry.crc32:
            raise PreparationError(f'ZIP integrity mismatch for selected member: {name}')
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + '.partial')
        partial.write_bytes(data)
        digest = _read_and_verify_existing(partial, entry)
        partial.replace(output)
        evidence[name] = _member_evidence(entry, output, digest, 'written')
    return evidence


def _member_evidence(
    entry: RemoteZipEntry, path: Path, digest: str, disposition: str,
) -> dict[str, Any]:
    return {
        'member': entry.name,
        'path': str(path),
        'compressed_size': entry.compressed_size,
        'uncompressed_size': entry.uncompressed_size,
        'crc32': f'{entry.crc32:08x}',
        'raw_sha256': digest,
        'disposition': disposition,
    }


def extract_range_members(url: str, index: RemoteZipIndex, members: Sequence[str], destination: Path) -> dict[str, str]:
    '''Compatibility wrapper returning only raw SHA-256 digests.'''

    return {
        name: row['raw_sha256']
        for name, row in extract_range_members_evidence(
            url, index, members, destination
        ).items()
    }
