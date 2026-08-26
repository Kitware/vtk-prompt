"""Launch a local vtk-mcp server as a subprocess for embedded use.

Lets ``vtk-prompt --embed-mcp`` provide context-enhanced generation without a
separately started vtk-mcp server (docker compose, manual
``vtk-mcp --transport http``, etc.).
"""

from __future__ import annotations

import contextlib
import importlib.util
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

from . import get_logger
from .vtk_mcp_client import check_mcp_available

logger = get_logger(__name__)

DEFAULT_STARTUP_TIMEOUT = 180.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def embedded_mcp_server(startup_timeout: float = DEFAULT_STARTUP_TIMEOUT) -> Iterator[str]:
    """Spawn a local vtk-mcp HTTP server for the duration of the ``with`` block.

    Yields the server's base URL. Raises RuntimeError if vtk-mcp isn't
    installed or doesn't become ready within ``startup_timeout`` seconds
    (the first run can be slow: vtk-mcp downloads its knowledge/embedding
    artifacts on first use).
    """
    if importlib.util.find_spec("vtk_mcp") is None:
        raise RuntimeError(
            "vtk-mcp is not installed. Install it with: pip install 'vtk-prompt[embedded-mcp]'"
        )

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    logger.info("Starting embedded vtk-mcp on %s", url)
    proc = subprocess.Popen(
        [sys.executable, "-m", "vtk_mcp", "--transport", "http", "--port", str(port)]
    )

    try:
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"embedded vtk-mcp exited early with code {proc.returncode}")
            if check_mcp_available(url):
                logger.info("Embedded vtk-mcp ready at %s", url)
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"embedded vtk-mcp did not become ready within {startup_timeout}s")

        yield url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
