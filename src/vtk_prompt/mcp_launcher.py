"""Launch a local vtk-mcp server as a subprocess for embedded use.

Lets ``vtk-prompt --embed-mcp`` provide context-enhanced generation without a
separately started vtk-mcp server (docker compose, manual
``vtk-mcp --transport http``, etc.). Talks to it over stdio, since vtk-prompt
owns the subprocess directly: no port to pick, no HTTP round-trip.
"""

from __future__ import annotations

import contextlib
import importlib.util
import subprocess
import sys
from collections.abc import Iterator

from . import get_logger
from .vtk_mcp_client import VTKMCPClient

logger = get_logger(__name__)

DEFAULT_STARTUP_TIMEOUT = 180.0


@contextlib.contextmanager
def embedded_mcp_server(
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    knowledge_artifact: str | None = None,
    vtk_version: str | None = None,
) -> Iterator[VTKMCPClient]:
    """Spawn a local vtk-mcp stdio server for the duration of the ``with`` block.

    ``knowledge_artifact`` points vtk-mcp at a local vtk-knowledge JSONL file,
    skipping its auto-download; ``vtk_version`` selects which version to fetch
    from ghcr.io when no local artifact is given.

    Yields a connected VTKMCPClient. Raises RuntimeError if vtk-mcp isn't
    installed or doesn't complete its handshake within ``startup_timeout``
    seconds (the first run can be slow: vtk-mcp downloads its knowledge/
    embedding artifacts on first use, before it starts serving).
    """
    if importlib.util.find_spec("vtk_mcp") is None:
        raise RuntimeError(
            "vtk-mcp is not installed. Install it with: pip install 'vtk-prompt[bundle-mcp]'"
        )

    argv = [sys.executable, "-m", "vtk_mcp"]
    if knowledge_artifact:
        argv += ["--knowledge-artifact", knowledge_artifact]
    if vtk_version:
        argv += ["--vtk-version", vtk_version]

    logger.info("Starting embedded vtk-mcp (stdio)")
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        client = VTKMCPClient(base_url=None, process=proc, startup_timeout=startup_timeout)
        if not client.ready:
            if proc.poll() is not None:
                raise RuntimeError(f"embedded vtk-mcp exited early with code {proc.returncode}")
            raise RuntimeError(f"embedded vtk-mcp did not become ready within {startup_timeout}s")

        logger.info("Embedded vtk-mcp ready")
        yield client
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
