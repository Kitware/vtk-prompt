"""
File Handlers Module.

This module provides file handling utilities for the VTK Prompt UI application,
including JavaScript loading and file operations.
"""

from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from typing import Any


def load_js(server: Any) -> None:
    """Load JavaScript utilities for VTK Prompt UI.

    The utils.js URL carries a hash of the file contents. Without it the browser
    caches it indefinitely (the path has no version in it), so edits are
    silently ignored until someone clears the cache by hand.

    Also aliases trame-code's serve directory at the unversioned path. Upstream
    mounts it as ``__trame_code_<version>`` to defeat browser caching, but the
    bundled JS still builds worker URLs as ``__trame_code/monacoeditorwork/*``,
    so those 404 and Monaco falls back to running worker code on the main
    thread ("might cause UI freezes") plus a warning per editor mount. Serving
    the same directory at both paths fixes it without patching the client.
    """
    js_file = Path(__file__).parent.parent / "utils.js"
    try:
        digest = sha256(js_file.read_bytes()).hexdigest()[:12]
    except OSError:
        digest = "dev"

    serve: dict[str, str] = {"vtk_prompt": str(js_file.parent)}
    # Located via importlib rather than imported: trame_code ships no type
    # stubs, and only its directory is needed here.
    spec = find_spec("trame_code")
    if spec is not None and spec.origin:
        worker_dir = Path(spec.origin).parent / "module" / "serve"
        if worker_dir.is_dir():
            serve["__trame_code"] = str(worker_dir)

    server.enable_module(
        {
            "serve": serve,
            "scripts": [f"vtk_prompt/{js_file.name}?v={digest}"],
        }
    )
