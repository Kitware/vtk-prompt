"""TextMate language definition for Python (grammar + language configuration).

Vendored from Kitware/trame-code's example so the Monaco editor gets proper
Python syntax highlighting, auto-closing pairs, comment toggling, and folding,
rather than Monaco's basic built-in Python mode.
"""

from pathlib import Path

CONFIGURATION = Path(__file__).with_name("python.config.json").read_text()
GRAMMAR = Path(__file__).with_name("python.grammar.json").read_text()
GRAMMAR_TYPE = "json"


def register_lang(config: dict) -> None:
    """Append the Python language, grammar, and config into a textmate bundle."""
    config["languages"].append(
        dict(
            id="python",
            extensions=[".py", ".pyw", ".pyi"],
            aliases=["Python", "py"],
            filenames=[],
            firstLine=r"^#!\s*/?.*\bpython[0-9.-]*\b",
        )
    )
    config["configs"]["python"] = CONFIGURATION
    config["grammars"]["source.python"] = {
        "language": "python",
        "content": (GRAMMAR_TYPE, GRAMMAR),
    }
