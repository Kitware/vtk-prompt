"""In-process Python/VTK completion and hover backed by jedi.

This runs inside the same trame process as the UI (no separate server). The UI
exposes ``complete_python`` / ``hover_python`` through trame triggers that the
Monaco editor's providers call over the existing websocket.

Both use ``jedi.Interpreter`` against a live namespace, so completion and hover
resolve not only ``vtk`` (with real docstrings) but also the objects injected
into the generated code's exec scope, such as ``renderer`` and
``render_window``. Call :func:`register_runtime_objects` once those exist so the
editor can complete e.g. ``renderer.AddActor`` and hover their docstrings.
"""

from __future__ import annotations

from . import get_logger

logger = get_logger(__name__)

# jedi is imported lazily so importing this module never hard-fails if jedi is
# missing; completion/hover just return nothing in that case.
try:
    import jedi  # type: ignore

    _JEDI_OK = True
except Exception:  # pragma: no cover - jedi is a declared dependency
    _JEDI_OK = False
    logger.warning("jedi not available; Python code completion disabled")

# A live namespace jedi.Interpreter resolves names against (without executing
# the user's code). Seeded with vtk; runtime objects are registered as they are
# created so the editor can complete/hover them too.
_NS: dict = {}
try:
    exec("import vtk\nimport vtkmodules.all", _NS)  # noqa: S102 - trusted, our own code
except Exception:  # pragma: no cover
    pass


def register_runtime_objects(**objects) -> None:
    """Expose live objects (e.g. ``renderer=...``) to completion and hover.

    The names become resolvable in the editor exactly as they are in the
    generated code's exec scope, so ``renderer.`` lists the live vtkRenderer's
    methods and hover shows their docstrings. Safe to call repeatedly.
    """
    _NS.update(objects)


def complete_python(code: str, line: int, column: int, limit: int = 500) -> list[dict]:
    """Return completion candidates for ``code`` at 1-based ``line`` / 0-based ``column``.

    Each candidate is a dict: {"label", "kind", "detail"} ready for the client
    to map onto Monaco completion items. Never raises; returns [] on any error.

    The limit is generous because Monaco caches the suggestion list returned at
    the trigger (e.g. right after ".") and filters it client-side as the user
    keeps typing. A small cap would silently hide any member outside the first N,
    even when the user types its exact name. VTK classes can expose a few hundred
    members (including inherited ones), so the cap must comfortably exceed that.
    """
    if not _JEDI_OK or not isinstance(code, str):
        return []
    try:
        completions = jedi.Interpreter(code, [_NS]).complete(line, column)
    except Exception as exc:  # jedi can raise on malformed/partial input
        logger.debug("jedi completion error: %s", exc)
        return []

    out: list[dict] = []
    for c in completions[:limit]:
        try:
            detail = (c.description or "")[:80]
        except Exception:
            detail = ""
        out.append({"label": c.name, "kind": c.type, "detail": detail})
    return out


def _doc_prose(doc: str, name: str) -> str:
    """Strip signature / C++ lines from a docstring, leaving the prose."""
    kept = []
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.startswith(name + "(") or stripped.startswith("C++:"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def hover_python(code: str, line: int, column: int) -> dict | None:
    """Return hover info (signature + docstring) for the symbol at the cursor.

    Uses jedi.Interpreter against the live namespace so VTK (and registered
    runtime objects) resolve with their real docstrings. 1-based ``line``,
    0-based ``column``. Never raises.
    """
    if not _JEDI_OK or not isinstance(code, str):
        return None
    try:
        defs = jedi.Interpreter(code, [_NS]).help(line, column)
    except Exception as exc:
        logger.debug("jedi hover error: %s", exc)
        return None
    if not defs:
        return None
    d = defs[0]
    try:
        signatures = [s.to_string() for s in d.get_signatures()]
    except Exception:
        signatures = []
    try:
        doc = d.docstring() or ""
    except Exception:
        doc = ""
    if not signatures and not doc:
        return None
    return {
        "name": d.name,
        "type": d.type,
        "signatures": signatures,
        "prose": _doc_prose(doc, d.name),
    }
