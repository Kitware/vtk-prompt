"""
Conversation Controllers Module.

This module provides controller functions for conversation navigation, file handling,
and conversation state management in the VTK Prompt UI.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

from .. import get_logger

logger = get_logger(__name__)

EXPLAIN_RENDERER = (
    "# renderer is a vtkRenderer injected by this webapp"
    + "\n"
    + "# Use your own vtkRenderer in your application"
)
EXTRA_INSTRUCTIONS_TAG = "</extra_instructions>"


def on_conversation_file_data_change(
    app: Any, conversation_object: Optional[dict[str, Any]], **_: Any
) -> None:
    """Handle conversation file data changes and load conversation history."""
    invalid = (
        conversation_object is None
        or conversation_object.get("type") != "application/json"
        or Path(conversation_object.get("name", "")).suffix != ".json"
        or not conversation_object.get("content")
    )

    if not invalid and conversation_object is not None:
        loaded_conversation = json.loads(conversation_object["content"])

        # Append loaded conversation to existing conversation instead of overwriting
        if app.state.conversation is None:
            app.state.conversation = []

        # Extend the existing conversation with the loaded one
        app.state.conversation.extend(loaded_conversation)
        app.state.conversation_file = conversation_object["name"]
        app.prompt_client.update_conversation(loaded_conversation, app.state.conversation_file)

        _process_loaded_conversation(app)
    else:
        app.state.conversation_file = None

    if not invalid and app.state.auto_run_conversation_file:
        app._conversation_loading = True
        app.generate_code()


def navigate_conversation_left(app: Any) -> None:
    """Navigate to previous conversation pair."""
    if not app.state.conversation_navigation:
        return

    if app.state.conversation_index >= 0:
        app.state.conversation_index -= 1
        if app.state.conversation_index >= 0:
            _process_conversation_pair(app)
        _update_navigation_state(app)


def navigate_conversation_right(app: Any) -> None:
    """Navigate to next conversation pair."""
    if not app.state.conversation_navigation:
        return

    nav_length = len(app.state.conversation_navigation)
    if app.state.conversation_index < nav_length:
        app.state.conversation_index += 1
        if app.state.conversation_index < nav_length:
            # Still viewing history
            _process_conversation_pair(app)
        else:
            # Moved to "new entry" mode - clear only query text for new input
            app.state.query_text = ""
        _update_navigation_state(app)


def save_conversation(app: Any) -> str:
    """Save current conversation history as JSON string."""
    if hasattr(app, "prompt_client") and app.prompt_client is not None:
        return json.dumps(app.prompt_client.conversation, indent=2)
    return ""


def _parse_assistant_content(content: str) -> tuple[Optional[str], Optional[str]]:
    """Parse assistant message content for explanation and code."""
    try:
        explanation_match = re.findall(r"<explanation>(.*?)</explanation>", content, re.DOTALL)
        code_match = re.findall(r"<code>(.*?)</code>", content, re.DOTALL)

        if explanation_match and code_match:
            return explanation_match[0].strip(), code_match[0].strip()
        return None, None
    except Exception:
        return None, None


def build_conversation_navigation(app: Any) -> None:
    """Build list of conversation pairs (user message + assistant response) for navigation."""
    if not app.state.conversation:
        app.state.conversation_navigation = []
        app.state.conversation_index = 0
        _update_navigation_state(app)
        return

    pairs = []
    current_user = None

    for message in app.state.conversation:
        if message.get("role") == "user":
            current_user = message
        elif message.get("role") == "assistant" and current_user:
            pairs.append({"user": current_user, "assistant": message})
            current_user = None

    app.state.conversation_navigation = pairs
    # Reset index to last pair if we have pairs
    if pairs:
        app.state.conversation_index = len(pairs) - 1
    else:
        app.state.conversation_index = 0

    _update_navigation_state(app)


def _update_navigation_state(app: Any) -> None:
    """Update navigation button states based on current position."""
    nav_length = len(app.state.conversation_navigation)

    # Update navigation buttons
    app.state.can_navigate_left = nav_length > 0 and app.state.conversation_index > 0
    app.state.can_navigate_right = nav_length > 0 and app.state.conversation_index < nav_length

    # Update viewing mode - we're viewing history if not at the "new entry" position
    app.state.is_viewing_history = nav_length > 0 and app.state.conversation_index < nav_length


def sync_with_prompt_client(app: Any) -> None:
    """Sync conversation navigation with prompt client conversation."""
    if app.prompt_client and app.prompt_client.conversation:
        app.state.conversation = app.prompt_client.conversation
        build_conversation_navigation(app)


def _process_conversation_pair(app: Any, pair_index: Optional[int] = None) -> None:
    """Process a specific conversation pair by index."""
    if not app.state.conversation_navigation:
        return

    if pair_index is None:
        pair_index = app.state.conversation_index

    if pair_index < 0 or pair_index >= len(app.state.conversation_navigation):
        return

    pair = app.state.conversation_navigation[pair_index]

    # Process assistant message for explanation and code
    assistant_content = pair["assistant"].get("content", "")
    explanation, code = _parse_assistant_content(assistant_content)

    if explanation and code:
        # Set explanation and code in UI state
        app.state.generated_explanation = explanation
        app.state.generated_code = EXPLAIN_RENDERER + "\n" + code

        # Execute the code to display visualization
        app._execute_with_renderer(code)

    # Process user message for query text
    user_content = pair["user"].get("content", "").strip()
    if EXTRA_INSTRUCTIONS_TAG in user_content:
        parts = user_content.split(EXTRA_INSTRUCTIONS_TAG, 1)
        query_text = parts[1].strip() if len(parts) > 1 else user_content
    else:
        query_text = user_content

    app.state.query_text = query_text


def _process_loaded_conversation(app: Any) -> None:
    """Process loaded conversation file."""
    if not app.state.conversation:
        return

    # Build navigation pairs and process the latest one
    build_conversation_navigation(app)
    _process_conversation_pair(app)
