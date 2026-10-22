"""Suggest corrected ``vtkmodules`` imports for code that fails to import.

VTK's Python classes are spread over many ``vtkmodules`` submodules, and a
model that knows the class name often guesses the wrong one. The fix does not
have to be a guess: an imported class reports where it actually lives, so the
correct import can be derived rather than matched approximately. Only a
misspelled class name falls back to fuzzy matching.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

# "cannot import name 'vtkFoo' from 'vtkmodules.vtkBar'"
_CANNOT_IMPORT = re.compile(
    r"cannot import name ['\"](?P<name>\w+)['\"](?: from ['\"](?P<module>[\w.]+)['\"])?"
)
# "No module named 'vtkmodules.vtkBar'"
_NO_MODULE = re.compile(r"No module named ['\"](?P<module>[\w.]+)['\"]")
_IMPORT_LINE = re.compile(r"^\s*from\s+(?P<module>[\w.]+)\s+import\s+(?P<names>.+?)\s*$")


def _namespace() -> dict:
    """Return the completion module's warm vtk namespace, importing if needed."""
    from .. import completion

    return completion.vtk_namespace()


def _module_of(class_name: str) -> str | None:
    """Return the vtkmodules submodule that defines ``class_name``."""
    ns = _namespace()
    for container in (ns.get("vtk"), ns.get("vtkmodules")):
        obj = getattr(container, class_name, None) if container is not None else None
        if obj is not None:
            module = getattr(obj, "__module__", None)
            if module and module.startswith("vtkmodules."):
                return module
    return None


def _known_class_names() -> list[str]:
    ns = _namespace()
    vtk_mod = ns.get("vtk")
    if vtk_mod is None:
        return []
    return [n for n in dir(vtk_mod) if n.startswith("vtk")]


def _import_line_for(code: str, name: str) -> tuple[int, str] | None:
    """Find the 1-based line importing ``name`` and its text."""
    for i, line in enumerate(code.splitlines(), start=1):
        match = _IMPORT_LINE.match(line)
        if match and name in [n.strip() for n in match.group("names").split(",")]:
            return i, line
    return None


def _rewrite_import(line: str, name: str, module: str, drop: str | None = None) -> str:
    """Move ``name`` to ``module``, keeping any other names on the line.

    ``drop`` removes a different name than the one being added, which is what a
    misspelling needs: the bad name goes, the real one arrives.
    """
    match = _IMPORT_LINE.match(line)
    if not match:
        return f"from {module} import {name}"
    indent = line[: len(line) - len(line.lstrip())]
    removed = {name, drop} if drop else {name}
    others = [n.strip() for n in match.group("names").split(",") if n.strip() not in removed]
    fixed = f"{indent}from {module} import {name}"
    if others:
        keep = f"{indent}from {match.group('module')} import {', '.join(others)}"
        return f"{keep}\n{fixed}"
    return fixed


def suggestions(code: str, error_message: str) -> list[dict]:
    """Return one-click import fixes for an import failure in ``code``.

    Each entry is {"name", "line", "old", "new", "reason"}. An empty list means
    the error was not an import problem, or the right import could not be
    determined.
    """
    if not code or not error_message:
        return []

    cannot = _CANNOT_IMPORT.search(error_message)
    if cannot:
        name = cannot.group("name")
        found = _import_line_for(code, name)
        if not found:
            return []
        line_no, line_text = found
        module = _module_of(name)
        if module and module != cannot.group("module"):
            return [
                {
                    "name": name,
                    "line": line_no,
                    "old": line_text,
                    "new": _rewrite_import(line_text, name, module),
                    "reason": f"{name} lives in {module}",
                }
            ]
        # The class does not exist under that name; offer the nearest real ones.
        out = []
        for candidate in difflib.get_close_matches(name, _known_class_names(), n=3, cutoff=0.7):
            target = _module_of(candidate)
            if not target:
                continue
            out.append(
                {
                    "name": name,
                    "line": line_no,
                    "old": line_text,
                    "new": _rewrite_import(line_text, candidate, target, drop=name),
                    "reason": f"did you mean {candidate}?",
                }
            )
        return out

    missing = _NO_MODULE.search(error_message)
    if missing and missing.group("module").startswith("vtkmodules"):
        bad_module = missing.group("module")
        out = []
        seen: set[str] = set()
        for i, line in enumerate(code.splitlines(), start=1):
            match = _IMPORT_LINE.match(line)
            if not match or match.group("module") != bad_module:
                continue
            for raw in match.group("names").split(","):
                name = raw.strip()
                module = _module_of(name)
                if not module or name in seen:
                    continue
                seen.add(name)
                out.append(
                    {
                        "name": name,
                        "line": i,
                        "old": line,
                        "new": _rewrite_import(line, name, module),
                        "reason": f"{name} lives in {module}",
                    }
                )
        return out

    return []


def apply_suggestion(code: str, suggestion: dict[str, Any]) -> str:
    """Return ``code`` with the suggested import line replaced."""
    old = suggestion.get("old")
    new = suggestion.get("new")
    if not old or not new:
        return code
    lines = code.splitlines()
    index = suggestion.get("line", 0) - 1
    if 0 <= index < len(lines) and lines[index] == old:
        lines[index] = new
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
    return code.replace(old, new, 1)
