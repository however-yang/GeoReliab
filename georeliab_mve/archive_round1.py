'''Streaming archive integrity and strict resumable download hardening.'''

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence
import urllib.request
import zipfile

from .preparation import PreparationError


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path, *, required_entries: Sequence[str] = (), expected_bytes: int | None = None, expected_sha256: str | None = None, allow_partial: bool = False) -> dict[str, object]:
    if (path.suffix == '.partial' and not allow_partial) or not path.is_file() or path.stat().st_size == 0:
        raise PreparationError(f'archive is missing or partial: {path}')
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise PreparationError(f'archive length mismatch: {path}')
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise PreparationError(f'archive verification failed: {path}: {exc}') from exc
    if bad is not None:
        raise PreparationError(f'archive CRC failure in {path}: {bad}')
    missing = [entry for entry in required_entries if entry not in names]
    if missing:
        raise PreparationError(f'archive missing required entries: {missing}')
    digest = _stream_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise PreparationError(f'archive SHA-256 mismatch: {path}')
    return {'path': str(path), 'sha256': digest, 'entries': len(names), 'bytes': path.stat().st_size}


def download_archive(url: str, destination: Path, *, dry_run: bool = False, expected_bytes: int | None = None, expected_sha256: str | None = None) -> Path:
    if dry_run:
        return destination.with_suffix(destination.suffix + '.partial')
    if destination.exists():
        # A complete destination is immutable unless it proves its frozen identity.
        verify_archive(destination, expected_bytes=expected_bytes, expected_sha256=expected_sha256)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        if offset:
            content_range = response.headers.get('Content-Range', '')
            if response.status != 206 or not content_range.startswith(f'bytes {offset}-'):
                raise PreparationError('resume requires HTTP 206 with matching Content-Range; refusing server-ignored append')
        elif response.status not in (200, 206):
            raise PreparationError(f'download returned unexpected HTTP status: {response.status}')
        with partial.open('ab' if offset else 'wb') as handle:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                handle.write(block)
    verify_archive(partial, expected_bytes=expected_bytes, expected_sha256=expected_sha256, allow_partial=True)
    partial.replace(destination)
    return destination
