"""Registry for user-uploaded custom data files.

Complements ``resolver.py`` (which fetches known VTK sample data by name) by
letting users supply their own files. Uploaded files are written to a local
cache directory and can then be referenced by bare filename in generated code,
exactly like sample data. User uploads take precedence over sample data of the
same name.
"""

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import cast

# Characters allowed in a stored filename; anything else is replaced.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _uploads_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    directory = root / "vtk-prompt" / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_name(filename: str) -> str:
    """Reduce an arbitrary filename to a bare, filesystem-safe basename."""
    name = os.path.basename((filename or "").strip())
    name = _SAFE_NAME_RE.sub("_", name)
    return name.lstrip(".") or "upload"


def _as_bytes(content: object) -> bytes:
    """Coerce uploaded file content (bytes, bytearray, str, or ints) to bytes."""
    if isinstance(content, bytes):
        return content
    if isinstance(content, (bytearray, memoryview)):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8", "surrogateescape")
    if isinstance(content, Iterable):
        try:
            return bytes(cast(Iterable[int], content))  # e.g. a list of byte values
        except (TypeError, ValueError):
            return b""
    return b""


def add_upload(filename: str, content: object) -> str:
    """Store uploaded content under its sanitized basename; return the path."""
    dest = _uploads_dir() / _safe_name(filename)
    with open(dest, "wb") as handle:
        handle.write(_as_bytes(content))
    return str(dest)


def uploaded_names() -> list[str]:
    """Sorted basenames of all uploaded files currently available."""
    directory = _uploads_dir()
    return sorted(p.name for p in directory.iterdir() if p.is_file())


def uploaded_path(name: str) -> str | None:
    """Local path for an uploaded file by basename, or None if absent."""
    name = os.path.basename((name or "").strip())
    if not name:
        return None
    path = _uploads_dir() / name
    return str(path) if path.is_file() else None


def remove_upload(name: str) -> bool:
    """Delete one uploaded file by basename. Returns True if it existed."""
    path = uploaded_path(name)
    if not path:
        return False
    Path(path).unlink(missing_ok=True)
    return True


def clear_uploads() -> None:
    """Remove all uploaded files."""
    for p in _uploads_dir().iterdir():
        if p.is_file():
            p.unlink(missing_ok=True)
