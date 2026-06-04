"""Startup discovery of a default config file and .env files for the UI.

Lets ``vtk-prompt-ui`` run with no arguments and still pick up a saved
configuration and environment variables (e.g. API tokens), instead of
requiring ``--prompt-file`` and an inline ``OPENAI_API_KEY=...`` prefix.
"""

import os
from pathlib import Path

CONFIG_ENV_VAR = "VTK_PROMPT_CONFIG"
APP_DIR_NAME = "vtk-prompt"


def _config_home() -> Path:
    """Return the per-user config directory (XDG-aware)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / APP_DIR_NAME


def discover_config_file() -> str | None:
    """Return the first existing default config path, or None.

    Search order:
      1. ``$VTK_PROMPT_CONFIG`` (explicit path)
      2. ``./vtk-prompt.yml`` / ``./vtk-prompt.yaml`` (current directory)
      3. ``~/.config/vtk-prompt/config.yml`` / ``config.yaml``
    """
    explicit = os.environ.get(CONFIG_ENV_VAR)
    if explicit and Path(explicit).is_file():
        return explicit

    candidates = [
        Path.cwd() / "vtk-prompt.yml",
        Path.cwd() / "vtk-prompt.yaml",
        _config_home() / "config.yml",
        _config_home() / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def load_dotenv_files() -> list[str]:
    """Load simple ``KEY=VALUE`` pairs from .env files into ``os.environ``.

    Existing environment variables are never overwritten. Returns the list of
    files actually loaded. Search order:
      1. ``~/.config/vtk-prompt/.env``
      2. ``./.env`` (current directory)
    """
    loaded: list[str] = []
    for path in (_config_home() / ".env", Path.cwd() / ".env"):
        if not path.is_file():
            continue
        try:
            for raw in path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, val)
            loaded.append(str(path))
        except OSError:
            continue
    return loaded
