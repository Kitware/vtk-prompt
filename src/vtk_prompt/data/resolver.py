"""Resolve named VTK example datasets to local file paths.

VTK examples reference data files by bare name (for example ``cow.g``). The bytes
live in VTK's content-addressed ExternalData store; each file has a
``<path>.sha512`` pointer somewhere in a VTK data tree. This module builds a
``basename -> sha512`` index from a local VTK data tree (pointed at by the
``VTK_PROMPT_DATA_ROOT`` environment variable), then resolves a name by fetching
the blob from the ExternalData store, caching it locally and verifying its
checksum. That lets generated example-style code which reads such files run.
"""

import hashlib
import logging
import os
import re
import shutil
import urllib.request
from pathlib import Path
from urllib.error import URLError

logger = logging.getLogger(__name__)

_STORE_URL = "https://data.kitware.com/api/v1/file/hashsum/sha512/{digest}/download"
_DATA_ROOT_ENV = "VTK_PROMPT_DATA_ROOT"
_POINTER_SUFFIX = ".sha512"

_index_cache: dict[str, str] | None = None


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    directory = root / "vtk-prompt" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _data_root() -> Path | None:
    root = os.environ.get(_DATA_ROOT_ENV)
    if root and Path(root).is_dir():
        return Path(root)
    return None


def _build_index(root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(root):
        for filename in files:
            if not filename.endswith(_POINTER_SUFFIX):
                continue
            name = filename[: -len(_POINTER_SUFFIX)]
            if name in index:
                continue  # first occurrence wins on duplicate basenames
            try:
                digest = (Path(dirpath) / filename).read_text().strip()
            except OSError:
                continue
            if digest:
                index[name] = digest
    return index


def _load_index() -> dict[str, str]:
    global _index_cache
    if _index_cache is None:
        root = _data_root()
        _index_cache = _build_index(root) if root else {}
    return _index_cache


def _sha512(path: Path) -> str:
    digest = hashlib.sha512()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(digest: str, dest: Path) -> bool:
    url = _STORE_URL.format(digest=digest)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out)
        return True
    except (OSError, URLError) as exc:
        logger.warning("Could not download %s: %s", dest.name, exc)
        dest.unlink(missing_ok=True)
        return False


def resolve(name: str) -> str | None:
    """Return a local path for a named dataset, downloading and caching as needed.

    Returns None if the name is unknown (not in the data index) or the download
    or checksum verification fails.
    """
    name = os.path.basename((name or "").strip())
    if not name:
        return None
    digest = _load_index().get(name)
    cached = _cache_dir() / name

    if cached.exists() and (not digest or _sha512(cached) == digest):
        return str(cached)
    if not digest:
        return None

    if not _download(digest, cached):
        return None
    if _sha512(cached) != digest:
        logger.warning("Checksum mismatch for %s; discarding", name)
        cached.unlink(missing_ok=True)
        return None
    return str(cached)


def available_names() -> list[str]:
    """Return the sorted list of dataset names known to the resolver."""
    return sorted(_load_index().keys())


def has_data_root() -> bool:
    """Whether a local VTK data tree is configured (enables name resolution)."""
    return _data_root() is not None


# Match single- or double-quoted string literals with no embedded quote/newline.
_LITERAL_RE = re.compile(r"([\'\"])([^\'\"\n]+?)\1")


def stage_code(code: str) -> str:
    """Rewrite bare data-file references in code to resolved local paths.

    Only string literals that are a bare filename (no directory separator) whose
    basename is a known dataset are rewritten, so example code like
    ``reader.SetFileName('cow.g')`` runs against the fetched file. Explicit paths
    and unrelated strings are left untouched.
    """
    index = _load_index()
    if not index or not code:
        return code

    def _replace(match: "re.Match[str]") -> str:
        quote, value = match.group(1), match.group(2)
        if ("/" in value) or ("\\" in value):
            return match.group(0)
        if value in index:
            path = resolve(value)
            if path:
                return f"{quote}{path}{quote}"
        return match.group(0)

    return _LITERAL_RE.sub(_replace, code)


def referenced(code: str) -> list[str]:
    """Return dataset names referenced as bare string literals in code (no fetch)."""
    index = _load_index()
    if not index or not code:
        return []
    found: list[str] = []
    for match in _LITERAL_RE.finditer(code):
        value = match.group(2)
        if ("/" in value) or ("\\" in value):
            continue
        if value in index and value not in found:
            found.append(value)
    return found


def artifacts(code: str) -> list[dict]:
    """Return data artifacts referenced by code (name, cache path, fetched flag).

    Pure lookup (no download); used to surface what files the current code pulls
    in and where they live.
    """
    cache = _cache_dir()
    result: list[dict] = []
    for name in referenced(code):
        path = cache / name
        result.append({"name": name, "path": str(path), "cached": path.exists()})
    return result
