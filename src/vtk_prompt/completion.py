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

import re
from functools import lru_cache
from types import ModuleType

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


def warm_up() -> None:
    """Prime jedi's analysis of the vtk module in a background thread.

    The first completion against ``vtk.`` triggers jedi to analyse the whole
    module, which takes several seconds; subsequent calls are ~10x faster
    because jedi caches that work. Doing it once at startup means the user's
    first real completion is fast enough that Monaco does not time out and
    close the popup. Never raises.
    """
    if not _JEDI_OK:
        return

    def _run() -> None:
        try:
            jedi.Interpreter("import vtk\nvtk.", [_NS]).complete(2, 4)
        except Exception as exc:  # warm-up is best-effort
            logger.debug("jedi warm-up error: %s", exc)

    import threading

    threading.Thread(target=_run, name="jedi-warmup", daemon=True).start()


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

    # Cheap path first, same reasoning as hover: a resolvable VTK receiver is
    # answered from docstrings without waiting on jedi's inference.
    cls = _resolve_receiver(code, line, column, allow_jedi=False)
    if cls is not None:
        members = _members_of(cls, limit)
        if members:
            return members

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

    if not out:
        # jedi gives nothing after a VTK call ("GetPointIds()."), so retry with
        # jedi allowed to infer the chain's root.
        cls = _resolve_receiver(code, line, column, allow_jedi=True)
        if cls is not None:
            return _members_of(cls, limit)
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


# --------------------------------------------------------------------------
# Docstring-based return-type resolution
#
# jedi cannot infer through VTK's C-extension methods. It *shows* the return
# type because it parses the docstring, but it does not use it for inference, so
# any chained expression dies at the first call: `tetra.GetPointIds().SetId` has
# no completions and no hover, even though `tetra.GetPointIds` resolves fine.
#
# VTK's docstrings do carry the type ("GetPointIds(self) -> vtkIdList"), so the
# chain can be walked manually: resolve the root with jedi, then step through
# each `.Method()` by reading the annotation and looking the class up in the vtk
# namespace. Only VTK object returns resolve, which is the intent - `-> int` or
# `-> None` correctly yields nothing to chain from.
# --------------------------------------------------------------------------

# "-> vtkIdList", "-> ('vtkIdList', ...)", "-> None"
_RETURN_ANNOTATION = re.compile(r"->\s*\(?\s*'?([A-Za-z_][\w.]*)'?")
# Trailing `name.Method().Method()` chain immediately left of the cursor. The
# final identifier is optional: completion fires straight after the "." with
# nothing typed, while hover sits inside a name that is already there.
_CHAIN = re.compile(
    r"([A-Za-z_]\w*)\s*((?:\.\s*[A-Za-z_]\w*\s*\([^()]*\)\s*)+)\.\s*\w*$"
)
_CHAIN_STEP = re.compile(r"\.\s*([A-Za-z_]\w*)\s*\([^()]*\)")
# `name.token` with no intervening call - the direct-attribute case.
_DIRECT = re.compile(r"([A-Za-z_]\w*)\s*\.\s*\w*$")


def _assigned_vtk_class(code: str, root: str) -> type | None:
    """Class from a literal ``root = vtk.vtkFoo()`` assignment in the source.

    Covers the dominant shape of generated VTK scripts without paying for jedi
    inference, which costs hundreds of milliseconds even with a warm cache.
    """
    m = re.search(
        rf"^\s*{re.escape(root)}\s*=\s*(?:[\w.]*\.)?(vtk[A-Za-z0-9_]*)\s*\(",
        code,
        re.MULTILINE,
    )
    return _vtk_class(m.group(1)) if m else None


def _live_class(root: str) -> type | None:
    """Class of an injected runtime object (``renderer``, ``render_window``).

    Modules and classes are excluded: ``vtk`` is in the namespace too, and
    ``type()`` of a module is useless here.
    """
    live = _NS.get(root)
    if live is None or isinstance(live, (type, ModuleType)):
        return None
    cls = type(live)
    return cls if cls.__name__.startswith("vtk") else None


@lru_cache(maxsize=4096)
def _return_class(cls: type, method: str) -> type | None:
    """Class returned by ``cls.method()`` per its docstring, or None.

    Cached because hover fires on mouse movement; the parsing is cheap but not
    free at that rate.
    """
    func = getattr(cls, method, None)
    doc = getattr(func, "__doc__", "") or ""
    for line in doc.splitlines():
        if not line.strip().startswith(method + "("):
            continue
        m = _RETURN_ANNOTATION.search(line)
        if not m:
            continue
        name = m.group(1).split(".")[-1]
        if not name.startswith("vtk"):
            return None  # int/float/None etc: nothing to chain from
        return _vtk_class(name)
    return None


def _vtk_class(name: str) -> type | None:
    """Look a VTK class up by bare name in the seeded namespace."""
    for container in (_NS.get("vtk"), _NS.get("vtkmodules")):
        obj = getattr(container, name, None)
        if isinstance(obj, type):
            return obj
    obj = _NS.get(name)
    return obj if isinstance(obj, type) else None


def _root_class(
    code: str, line: int, column: int, root: str, allow_jedi: bool
) -> type | None:
    """Class of the chain's root variable, cheapest source first.

    Injected runtime objects and literal ``x = vtk.vtkFoo()`` assignments are
    resolved without jedi, which is the whole point: jedi's inference is what
    makes hover slow enough for the editor to give up waiting.
    """
    cls = _live_class(root) or _assigned_vtk_class(code, root)
    if cls is not None or not allow_jedi or not _JEDI_OK:
        return cls
    try:
        for d in jedi.Interpreter(code, [_NS]).infer(line, column):
            obj = _vtk_class(d.name)
            if obj is not None:
                return obj
    except Exception as exc:
        logger.debug("jedi root inference error: %s", exc)
    return None


def _resolve_receiver(
    code: str, line: int, column: int, allow_jedi: bool = True
) -> type | None:
    """Class of the expression immediately left of the cursor, or None.

    Handles both ``tetra.GetPointIds().SetId`` (walking the chain by docstring
    return type) and plain ``renderer.AddActor``.
    """
    lines = code.splitlines()
    if line < 1 or line > len(lines):
        return None
    text = lines[line - 1]
    prefix = text[: column + 1]
    # Include the identifier the cursor sits inside, not just what precedes it.
    trailing = re.match(r"\w*", text[column + 1 :])
    prefix += trailing.group(0) if trailing else ""

    m = _CHAIN.search(prefix)
    steps_text = m.group(2) if m else ""
    if m is None:
        m = _DIRECT.search(prefix)
        if m is None:
            return None
    root = m.group(1)
    # Column of the root token, so jedi infers the variable and not the chain.
    cls = _root_class(
        code, line, prefix.index(root) + len(root), root, allow_jedi
    )
    if cls is None:
        return None
    for step in _CHAIN_STEP.findall(steps_text):
        cls = _return_class(cls, step)
        if cls is None:
            return None
    return cls


def _members_of(cls: type, limit: int) -> list[dict]:
    """Completion candidates for a resolved class, in jedi's output shape."""
    out: list[dict] = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(cls, name, None)
            doc = (getattr(attr, "__doc__", "") or "").splitlines()
            detail = next((ln.strip() for ln in doc if ln.strip()), "")[:80]
            kind = "function" if callable(attr) else "instance"
        except Exception:
            detail, kind = "", "instance"
        out.append({"label": name, "kind": kind, "detail": detail})
        if len(out) >= limit:
            break
    return out


def _hover_from_class(cls: type, name: str) -> dict | None:
    """Hover payload for ``cls.name``, matching hover_python's shape."""
    attr = getattr(cls, name, None)
    if attr is None:
        return None
    doc = getattr(attr, "__doc__", "") or ""
    if not doc:
        return None
    signatures = [
        ln.strip() for ln in doc.splitlines() if ln.strip().startswith(name + "(")
    ]
    return {
        "name": name,
        "type": "function" if callable(attr) else "instance",
        "signatures": signatures,
        "prose": _doc_prose(doc, name),
    }


def _token_at(code: str, line: int, column: int) -> str:
    """The identifier the cursor sits in or next to."""
    lines = code.splitlines()
    if line < 1 or line > len(lines):
        return ""
    text = lines[line - 1]
    start = column
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
        start -= 1
    end = column
    while end < len(text) and (text[end].isalnum() or text[end] == "_"):
        end += 1
    return text[start:end]


def hover_python(code: str, line: int, column: int) -> dict | None:
    """Return hover info (signature + docstring) for the symbol at the cursor.

    Uses jedi.Interpreter against the live namespace so VTK (and registered
    runtime objects) resolve with their real docstrings. 1-based ``line``,
    0-based ``column``. Never raises.
    """
    if not _JEDI_OK or not isinstance(code, str):
        return None

    # Try the docstring route before jedi. For a VTK receiver it answers in
    # ~1ms where jedi's help() takes hundreds of milliseconds warm and seconds
    # cold - long enough that the editor cancels the request and shows nothing,
    # which looked like "hover is broken" rather than "hover is slow".
    token = _token_at(code, line, column)
    if token:
        cls = _resolve_receiver(code, line, column, allow_jedi=False)
        if cls is not None:
            info = _hover_from_class(cls, token)
            if info is not None:
                return info

    try:
        defs = jedi.Interpreter(code, [_NS]).help(line, column)
    except Exception as exc:
        logger.debug("jedi hover error: %s", exc)
        return None
    if not defs:
        # Nothing from jedi either: retry the chain with jedi allowed to infer
        # the root, for receivers the cheap paths cannot see.
        cls = _resolve_receiver(code, line, column, allow_jedi=True)
        if cls is not None and token:
            return _hover_from_class(cls, token)
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
