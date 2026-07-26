"""TOML loading compatible with the frozen Python 3.10 runtimes.

Python 3.11 and newer use :mod:`tomllib` directly. The A100 environments are
frozen on Python 3.10 and intentionally cannot be mutated to install ``tomli``;
there the strict fallback below supports the TOML constructs used by the
project's committed protocol and deployment overlay.
"""

from __future__ import annotations

import re
from typing import Any, Iterator


class _FallbackTOMLDecodeError(ValueError):
    """Raised when the internal TOML subset parser rejects a document."""


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_DECIMAL_INT = re.compile(r"^[+-]?(?:0|[1-9](?:_?[0-9])*)$")
_HEX_INT = re.compile(r"^[+-]?0x[0-9A-Fa-f](?:_?[0-9A-Fa-f])*$")
_OCTAL_INT = re.compile(r"^[+-]?0o[0-7](?:_?[0-7])*$")
_BINARY_INT = re.compile(r"^[+-]?0b[01](?:_?[01])*$")
_FLOAT = re.compile(
    r"^[+-]?(?:"
    r"(?:0|[1-9](?:_?[0-9])*)\.[0-9](?:_?[0-9])*"
    r"(?:[eE][+-]?[0-9](?:_?[0-9])*)?"
    r"|(?:0|[1-9](?:_?[0-9])*)[eE][+-]?[0-9](?:_?[0-9])*"
    r")$"
)


def _fail(message: str, line: int | None = None) -> _FallbackTOMLDecodeError:
    suffix = f" (line {line})" if line is not None else ""
    return _FallbackTOMLDecodeError(message + suffix)


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif quote == "'":
            if char == "'":
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "#":
            return line[:index]
    return line


def _split_assignment(statement: str, line_number: int) -> tuple[str, str]:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(statement):
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif quote == "'":
            if char == "'":
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "=":
            key = statement[:index].strip()
            value = statement[index + 1 :].strip()
            if not key or not value:
                raise _fail("assignment requires a key and value", line_number)
            return key, value
    raise _fail("expected key = value assignment", line_number)


def _value_complete(value: str, line_number: int) -> bool:
    quote: str | None = None
    escaped = False
    square_depth = 0
    curly_depth = 0
    for char in value:
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
            continue
        if quote == "'":
            if char == "'":
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            curly_depth += 1
        elif char == "}":
            curly_depth -= 1
        if square_depth < 0 or curly_depth < 0:
            raise _fail("unbalanced value delimiter", line_number)
    return quote is None and square_depth == 0 and curly_depth == 0


def _logical_statements(text: str) -> Iterator[tuple[int, str]]:
    pending: list[str] = []
    start_line = 0
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw_line).strip()
        if not pending:
            if not line:
                continue
            if line.startswith("["):
                yield line_number, line
                continue
            pending = [line]
            start_line = line_number
        elif line:
            pending.append(line)
        statement = "\n".join(pending)
        _, value = _split_assignment(statement, start_line)
        if _value_complete(value, start_line):
            yield start_line, statement
            pending = []
            start_line = 0
    if pending:
        raise _fail("unterminated TOML value", start_line)


def _split_top_level(value: str, delimiter: str, line_number: int) -> list[str]:
    parts: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    square_depth = 0
    curly_depth = 0
    for index, char in enumerate(value):
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
            continue
        if quote == "'":
            if char == "'":
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            curly_depth += 1
        elif char == "}":
            curly_depth -= 1
        elif char == delimiter and square_depth == 0 and curly_depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    if quote is not None or square_depth != 0 or curly_depth != 0:
        raise _fail("unbalanced TOML expression", line_number)
    parts.append(value[start:].strip())
    return parts


def _parse_basic_string(value: str, line_number: int) -> str:
    if len(value) < 2 or not value.endswith('"'):
        raise _fail("unterminated basic string", line_number)
    content = value[1:-1]
    output: list[str] = []
    escapes = {
        "b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r",
        '"': '"', "\\": "\\",
    }
    index = 0
    while index < len(content):
        char = content[index]
        if char != "\\":
            if ord(char) < 0x20 and char != "\t":
                raise _fail("control character in basic string", line_number)
            output.append(char)
            index += 1
            continue
        index += 1
        if index >= len(content):
            raise _fail("unterminated escape sequence", line_number)
        code = content[index]
        if code in escapes:
            output.append(escapes[code])
            index += 1
            continue
        if code in ("u", "U"):
            width = 4 if code == "u" else 8
            digits = content[index + 1 : index + 1 + width]
            if len(digits) != width or not all(
                digit in "0123456789abcdefABCDEF" for digit in digits
            ):
                raise _fail("invalid Unicode escape", line_number)
            codepoint = int(digits, 16)
            if codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
                raise _fail("invalid Unicode scalar value", line_number)
            output.append(chr(codepoint))
            index += width + 1
            continue
        raise _fail(f"unsupported escape sequence \\{code}", line_number)
    return "".join(output)


def _parse_string(value: str, line_number: int) -> str:
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise _fail("unterminated literal string", line_number)
        content = value[1:-1]
        if "'" in content or "\n" in content or "\r" in content:
            raise _fail("invalid literal string", line_number)
        return content
    return _parse_basic_string(value, line_number)


def _parse_value(value: str, line_number: int) -> Any:
    value = value.strip()
    if not value:
        raise _fail("empty TOML value", line_number)
    if value[0] in ("'", '"'):
        return _parse_string(value, line_number)
    if value.startswith("["):
        if not value.endswith("]"):
            raise _fail("unterminated array", line_number)
        body = value[1:-1].strip()
        if not body:
            return []
        parts = _split_top_level(body, ",", line_number)
        if parts and parts[-1] == "":
            parts.pop()
        if any(part == "" for part in parts):
            raise _fail("empty array element", line_number)
        return [_parse_value(part, line_number) for part in parts]
    if value.startswith("{"):
        raise _fail("inline tables are not supported by the frozen parser", line_number)
    if value == "true":
        return True
    if value == "false":
        return False
    normalized = value.replace("_", "")
    if _DECIMAL_INT.fullmatch(value):
        return int(normalized, 10)
    if _HEX_INT.fullmatch(value):
        return int(normalized, 16)
    if _OCTAL_INT.fullmatch(value):
        return int(normalized, 8)
    if _BINARY_INT.fullmatch(value):
        return int(normalized, 2)
    if _FLOAT.fullmatch(value):
        return float(normalized)
    if value in ("inf", "+inf"):
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if value in ("nan", "+nan"):
        return float("nan")
    if value == "-nan":
        return -float("nan")
    raise _fail(f"unsupported TOML value: {value!r}", line_number)


def _parse_key_part(value: str, line_number: int) -> str:
    value = value.strip()
    if not value:
        raise _fail("empty TOML key", line_number)
    if value[0] in ("'", '"'):
        parsed = _parse_string(value, line_number)
        if not parsed:
            raise _fail("empty TOML key", line_number)
        return parsed
    if not _BARE_KEY.fullmatch(value):
        raise _fail(f"invalid bare key: {value!r}", line_number)
    return value


def _parse_key_path(value: str, line_number: int) -> tuple[str, ...]:
    return tuple(
        _parse_key_part(part, line_number)
        for part in _split_top_level(value, ".", line_number)
    )


def _descend(
    table: dict[str, Any], path: tuple[str, ...], line_number: int
) -> dict[str, Any]:
    current = table
    for part in path:
        value = current.get(part)
        if value is None:
            value = {}
            current[part] = value
        elif isinstance(value, list):
            if not value or not isinstance(value[-1], dict):
                raise _fail(f"cannot descend through array {part!r}", line_number)
            value = value[-1]
        elif not isinstance(value, dict):
            raise _fail(f"key {part!r} is already a scalar", line_number)
        current = value
    return current


def _loads_fallback(text: str) -> dict[str, Any]:
    """Parse the strict TOML subset used by committed GeoReliab config files."""

    if not isinstance(text, str):
        raise TypeError("toml_compat.loads() expects str, not bytes")
    document: dict[str, Any] = {}
    current = document
    explicit_tables: set[tuple[str, ...]] = set()
    for line_number, statement in _logical_statements(text):
        if statement.startswith("[["):
            if not statement.endswith("]]"):
                raise _fail("invalid array-of-tables header", line_number)
            path = _parse_key_path(statement[2:-2].strip(), line_number)
            if not path:
                raise _fail("empty array-of-tables header", line_number)
            parent = _descend(document, path[:-1], line_number)
            existing = parent.get(path[-1])
            if existing is None:
                existing = []
                parent[path[-1]] = existing
            if not isinstance(existing, list):
                raise _fail("array-of-tables conflicts with existing key", line_number)
            entry: dict[str, Any] = {}
            existing.append(entry)
            current = entry
            continue
        if statement.startswith("["):
            if not statement.endswith("]") or statement.startswith("[["):
                raise _fail("invalid table header", line_number)
            path = _parse_key_path(statement[1:-1].strip(), line_number)
            if not path:
                raise _fail("empty table header", line_number)
            if path in explicit_tables:
                raise _fail("duplicate table header", line_number)
            current = _descend(document, path, line_number)
            explicit_tables.add(path)
            continue
        key_text, value_text = _split_assignment(statement, line_number)
        key_path = _parse_key_path(key_text, line_number)
        parent = _descend(current, key_path[:-1], line_number)
        key = key_path[-1]
        if key in parent:
            raise _fail(f"duplicate key {key!r}", line_number)
        parent[key] = _parse_value(value_text, line_number)
    return document


try:
    import tomllib as _stdlib_tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on frozen A100 Python
    TOMLDecodeError = _FallbackTOMLDecodeError

    def loads(text: str) -> dict[str, Any]:
        return _loads_fallback(text)

    USING_STDLIB = False
else:
    TOMLDecodeError = _stdlib_tomllib.TOMLDecodeError
    loads = _stdlib_tomllib.loads
    USING_STDLIB = True


__all__ = ["TOMLDecodeError", "USING_STDLIB", "loads"]
