"""In-process Python/VTK completion backed by jedi.

This runs inside the same trame process as the UI (no separate server). The UI
exposes ``complete_python`` through a trame trigger ("jedi_complete") that the
Monaco editor's completion provider calls over the existing websocket.

VTK resolves because a small preamble (``import vtk``) is prepended before
analysis; jedi infers types through it, so e.g. ``vtkSphereSource().Set`` yields
the real ``Set*`` methods. The preamble line offset is added to the request line.
"""

from __future__ import annotations

from . import get_logger

logger = get_logger(__name__)

_PREAMBLE = "import vtk\nimport vtkmodules.all\n"
_PREAMBLE_LINES = _PREAMBLE.count("\n")

# jedi is imported lazily so importing this module never hard-fails if jedi is
# missing; completion just returns nothing in that case.
try:
    import jedi  # type: ignore

    _JEDI_OK = True
except Exception:  # pragma: no cover - jedi is a declared dependency
    _JEDI_OK = False
    logger.warning("jedi not available; Python code completion disabled")


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
        script = jedi.Script(code=_PREAMBLE + code)
        completions = script.complete(line + _PREAMBLE_LINES, column)
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


# A live namespace for hover introspection. jedi.Interpreter resolves names
# against these live modules (without executing the user's code), which yields
# VTK's real docstrings (e.g. "Set the radius of sphere. Default is 0.5.")
# rather than the signature-only text static analysis returns for compiled
# extension modules.
_HOVER_NS: dict = {}
try:
    exec("import vtk\nimport vtkmodules.all", _HOVER_NS)  # noqa: S102 - trusted, our own code
except Exception:  # pragma: no cover
    pass


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

    Uses jedi.Interpreter against a live VTK namespace so VTK docstrings include
    their real prose. 1-based ``line``, 0-based ``column``. Never raises.
    """
    if not _JEDI_OK or not isinstance(code, str):
        return None
    try:
        defs = jedi.Interpreter(code, [_HOVER_NS]).help(line, column)
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
