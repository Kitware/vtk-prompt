"""VTK-Prompt - CLI tool for generating VTK visualizations using LLMs.

This package provides tools to generate VTK Python code and XML files using
LLMs (Anthropic Claude, OpenAI GPT, or NVIDIA NIM models). It also includes
Retrieval-Augmented Generation (RAG) capabilities to improve code generation
by providing relevant examples from the VTK examples corpus.

Main components:
- vtk-prompt: Generate and run VTK Python code
- gen-vtk-file: Generate VTK XML files
- vtk-build-rag: Build a RAG database from VTK examples
- vtk-test-rag: Test the RAG database with queries
"""

import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("vtk-prompt")
except PackageNotFoundError:
    __version__ = "unknown"
__author__ = "Vicente Adolfo Bolea Sanchez"
__email__ = "vicente.bolea@kitware.com"


def setup_logging(level: str | None = None, log_file: str | None = None) -> None:
    """Configure logging for the vtk-prompt package.

    Args:
      level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
      log_file: Optional file path to write logs to
    """
    if level is None:
        level = os.environ.get("VTK_PROMPT_LOG_LEVEL", "INFO").upper()

    if log_file is None:
        log_file = os.environ.get("VTK_PROMPT_LOG_FILE")

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)

    # Configure root logger
    logger = logging.getLogger("vtk_prompt")
    logger.setLevel(getattr(logging, level, logging.INFO))

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    for handler in handlers:
        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.

    Args:
      name: Module name

    Returns:
      Configured logger instance
    """
    # Ensure logging is setup
    if not logging.getLogger("vtk_prompt").handlers:
        setup_logging()

    return logging.getLogger(f"vtk_prompt.{name}")


# Initialize logging on package import
setup_logging()
