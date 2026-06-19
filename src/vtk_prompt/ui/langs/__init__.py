"""Editor language definitions (TextMate grammars + configs) for trame-code."""

from . import python


def build_textmate() -> dict:
    """Build the textmate bundle passed to code.Editor's ``textmate`` prop."""
    config: dict = {"languages": [], "grammars": {}, "configs": {}}
    python.register_lang(config)
    return config


# Built once at import; serialized into trame state and sent to the client once.
PYTHON_TEXTMATE = build_textmate()
