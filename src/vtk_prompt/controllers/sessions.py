"""Session management: multiple conversations the user can switch between.

A session bundles one conversation's full state: the LLM message context, the
single code-version timeline (history + labels + position), the per-turn
checkpoints, plus metadata (title, timestamps, pinned). Exactly one session is
active at a time. Switching captures the live app state into the current session
and loads the target; "New" archives the current session and starts a fresh one,
so the rest of the list is preserved.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from ..utils.env_config import _config_home

logger = logging.getLogger(__name__)

# Keys written to / read from each session's JSON file.
_PERSIST_KEYS = (
    "id", "title", "created", "updated", "pinned", "messages",
    "code_history", "code_history_labels", "code_history_pos", "checkpoints",
)


def _sessions(app: Any) -> dict:
    """Backing store: id -> session dict (a plain attribute, not trame state)."""
    if not hasattr(app, "_session_store"):
        app._session_store = {}
    return app._session_store


def _new_session() -> dict:
    now = time.time()
    return {
        "id": uuid.uuid4().hex,
        "title": "New conversation",
        "created": now,
        "updated": now,
        "pinned": False,
        "messages": [],
        "code_history": [],
        "code_history_labels": [],
        "code_history_pos": -1,
        "checkpoints": [],
    }


def ensure_session(app: Any) -> dict:
    """Guarantee a current session exists; create the first one if needed."""
    sessions = _sessions(app)
    cur = getattr(app.state, "current_session_id", "") or ""
    if cur and cur in sessions:
        return sessions[cur]
    sess = _new_session()
    sessions[sess["id"]] = sess
    app.state.current_session_id = sess["id"]
    return sess


def current_session(app: Any) -> dict:
    """Return the active session object (creating one if needed)."""
    return ensure_session(app)


def _truncate(text: str, limit: int = 60) -> str:
    text = (text or "").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _maybe_title(app: Any, sess: dict) -> None:
    """Set a session's title from its first user prompt (once it has one)."""
    if sess["title"] not in ("", "New conversation"):
        return
    nav = app.state.conversation_navigation or []
    if not nav:
        return
    from .conversation import EXTRA_INSTRUCTIONS_TAG

    content = (nav[0].get("user", {}).get("content", "") or "").strip()
    if EXTRA_INSTRUCTIONS_TAG in content:
        content = content.split(EXTRA_INSTRUCTIONS_TAG, 1)[-1].strip()
    content = content.replace("Request:", "").strip()
    if content:
        sess["title"] = _truncate(content)


def capture_current_session(app: Any) -> None:
    """Snapshot the live app state into the current session object."""
    sess = current_session(app)
    client = getattr(app, "prompt_client", None)
    messages = list(getattr(client, "conversation", None) or app.state.conversation or [])
    sess["messages"] = messages
    sess["code_history"] = list(app.state.code_history or [])
    sess["code_history_labels"] = list(app.state.code_history_labels or [])
    sess["code_history_pos"] = app.state.code_history_pos
    sess["checkpoints"] = list(getattr(app, "_conversation_checkpoints", None) or [])
    _maybe_title(app, sess)
    _persist_session(sess)


def refresh_sessions_list(app: Any) -> None:
    """Rebuild the drawer-visible list: pinned first, then by the sort order."""
    sort_order = getattr(app.state, "history_sort_order", "newest") or "newest"
    recent_first = sort_order != "oldest"

    def _key(s: dict):
        updated = s.get("updated", 0)
        return (0 if s.get("pinned") else 1, -updated if recent_first else updated)

    ordered = sorted(_sessions(app).values(), key=_key)
    cur = getattr(app.state, "current_session_id", "") or ""
    app.state.sessions_list = [
        {
            "id": s["id"],
            "title": s["title"] or "New conversation",
            "pinned": s["pinned"],
            "active": s["id"] == cur,
        }
        for s in ordered
    ]


def _reset_live(app: Any) -> None:
    """Clear all live conversation/code state (the fresh-conversation hinge)."""
    client = getattr(app, "prompt_client", None)
    if client:
        client.conversation = []
        client.conversation_file = None
    app._conversation_checkpoints = []
    app.state.conversation = []
    app.state.conversation_navigation = []
    app.state.conversation_index = 0
    app.state.conversation_file = None
    app.state.query_text = ""
    app.state.generated_code = ""
    app.state.generated_explanation = ""
    app.state.code_history = []
    app.state.code_history_labels = []
    app.state.code_history_pos = -1


def load_session(app: Any, session_id: str, execute: bool = True) -> None:
    """Restore a session's saved state into the live app and render it.

    ``execute=False`` restores state without running the code (used at startup,
    before the render window and client are ready).
    """
    sessions = _sessions(app)
    if session_id not in sessions:
        return
    sess = sessions[session_id]
    app.state.current_session_id = session_id

    client = getattr(app, "prompt_client", None)
    if client:
        client.conversation = list(sess["messages"])
        client.conversation_file = None
    app.state.conversation = list(sess["messages"])
    app.state.conversation_file = None
    app.state.code_history = list(sess["code_history"])
    app.state.code_history_labels = list(sess["code_history_labels"])
    app.state.code_history_pos = sess["code_history_pos"]
    app._conversation_checkpoints = list(sess["checkpoints"])

    from .conversation import (
        _parse_assistant_content,
        _update_navigation_state,
        build_conversation_navigation,
    )

    build_conversation_navigation(app)
    _update_navigation_state(app)

    history = app.state.code_history or []
    pos = app.state.code_history_pos
    if history and 0 <= pos < len(history):
        app.state.generated_code = history[pos]
    else:
        app.state.generated_code = ""

    nav = app.state.conversation_navigation or []
    if nav:
        explanation, _ = _parse_assistant_content(
            nav[-1].get("assistant", {}).get("content", "")
        )
        app.state.generated_explanation = explanation or ""
    else:
        app.state.generated_explanation = ""
    app.state.query_text = ""

    if execute and app.state.generated_code:
        app._execute_with_renderer(app.state.generated_code)


def switch_session(app: Any, session_id: str) -> None:
    """Capture the current session, then load the requested one."""
    if session_id == (getattr(app.state, "current_session_id", "") or ""):
        return
    capture_current_session(app)
    load_session(app, session_id)
    refresh_sessions_list(app)


def new_session(app: Any) -> None:
    """Archive the current session and start a fresh empty one (keep the rest)."""
    capture_current_session(app)
    sess = _new_session()
    _sessions(app)[sess["id"]] = sess
    app.state.current_session_id = sess["id"]
    _reset_live(app)
    from .conversation import _update_navigation_state

    _update_navigation_state(app)
    refresh_sessions_list(app)


def touch_current_session(app: Any) -> None:
    """After a generation: bump recency, capture state, and refresh the list."""
    current_session(app)["updated"] = time.time()
    capture_current_session(app)
    refresh_sessions_list(app)


def toggle_pin_session(app: Any, session_id: str) -> None:
    """Pin or unpin a session so it sorts to the top of the Recents list."""
    sessions = _sessions(app)
    if session_id in sessions:
        sessions[session_id]["pinned"] = not sessions[session_id]["pinned"]
        _persist_session(sessions[session_id])
        refresh_sessions_list(app)


def _sessions_dir() -> Path:
    """Directory holding one JSON file per persisted session."""
    directory = _config_home() / "sessions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _session_path(session_id: str) -> Path:
    return _sessions_dir() / f"{session_id}.json"


def _persist_session(sess: dict) -> None:
    """Write a session to disk (skip empty ones so we do not litter files)."""
    if not sess.get("messages"):
        return
    try:
        data = {key: sess.get(key) for key in _PERSIST_KEYS}
        _session_path(sess["id"]).write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist session %s: %s", sess.get("id"), exc)


def _delete_session_file(session_id: str) -> None:
    try:
        path = _session_path(session_id)
        if path.exists():
            path.unlink()
    except OSError as exc:
        logger.warning("Could not delete session file %s: %s", session_id, exc)


def load_persisted_sessions(app: Any) -> None:
    """Load saved sessions from disk and open the most recently updated one.

    Called once at startup. Restores state only (no render); if nothing is
    saved, falls back to creating a fresh empty session.
    """
    store = _sessions(app)
    try:
        files = sorted(_sessions_dir().glob("*.json"))
    except OSError:
        files = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable session file %s: %s", path.name, exc)
            continue
        session_id = data.get("id")
        if not session_id:
            continue
        base = _new_session()
        for key in _PERSIST_KEYS:
            if key in data and data[key] is not None:
                base[key] = data[key]
        base["id"] = session_id
        store[session_id] = base

    if store:
        most_recent = max(store.values(), key=lambda s: s.get("updated", 0))
        app.state.current_session_id = most_recent["id"]
        load_session(app, most_recent["id"], execute=False)
    else:
        ensure_session(app)
    refresh_sessions_list(app)


def rename_session(app: Any, session_id: str, title: str) -> None:
    """Rename a conversation; ignore an all-whitespace title."""
    sessions = _sessions(app)
    if session_id not in sessions:
        return
    new_title = (title or "").strip()[:80]
    if not new_title:
        return
    sessions[session_id]["title"] = new_title
    _persist_session(sessions[session_id])
    refresh_sessions_list(app)


def delete_session(app: Any, session_id: str) -> None:
    """Delete a conversation; if it was active, open the next most recent."""
    sessions = _sessions(app)
    if session_id not in sessions:
        return
    was_current = session_id == (getattr(app.state, "current_session_id", "") or "")
    del sessions[session_id]
    _delete_session_file(session_id)

    if was_current:
        if sessions:
            most_recent = max(sessions.values(), key=lambda s: s.get("updated", 0))
            load_session(app, most_recent["id"])
        else:
            sess = _new_session()
            sessions[sess["id"]] = sess
            app.state.current_session_id = sess["id"]
            _reset_live(app)
            from .conversation import _update_navigation_state

            _update_navigation_state(app)
    refresh_sessions_list(app)
