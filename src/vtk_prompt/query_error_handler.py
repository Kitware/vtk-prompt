"""
Query Error Handler for VTK Prompt System.

This module provides error handling and correction hints for common VTK code generation
issues. It includes pattern matching for various Python and VTK-specific errors and
provides intelligent suggestions for fixing them.

The QueryErrorHandler class analyzes execution errors and generates retry queries with
specific guidance based on error patterns, helping the LLM learn from mistakes and
generate better code in subsequent attempts.

Classes:
    QueryErrorHandler: Main error handling and hint generation class

Example:
    >>> handler = QueryErrorHandler()
    >>> hint = handler.get_error_hint("AttributeError: 'vtkActor' has no attribute 'SetColour'")
    >>> retry_query = handler.build_retry_query(error, query, code, history)
"""

import re


class QueryErrorHandler:
    """Handles error patterns and provides correction hints."""

    # List of (regex_pattern, hint_template) pairs
    ERROR_HINTS = [
        # AttributeError
        (
            r"has no attribute '(\w+)'",
            "The object does not have an attribute '{0}'. "
            "Check for typos, use 'Set{0}' or 'Get{0}' if following naming conventions, "
            "or find an alternative method/property to achieve the desired effect.",
        ),
        # NameError
        (
            r"name '(\w+)' is not defined",
            "The name '{0}' is not defined. Ensure it is spelled correctly, "
            "imported, or created before use.",
        ),
        # ModuleNotFoundError
        (
            r"No module named '([\w\.]+)'",
            "The module '{0}' was not found. Ensure it is installed (`pip install {0}`) "
            "and imported with the correct name.",
        ),
        # ImportError (wrong import path)
        (
            r"cannot import name '(\w+)' from '([\w\.]+)'",
            "The object '{0}' cannot be imported from '{1}'. "
            "Verify the correct import path or library version.",
        ),
        # TypeError: wrong number of arguments
        (
            r"(\w+)\(\) takes (\d+) positional arguments but (\d+) were given",
            "The function or method '{0}' was called with {2} arguments but expects {1}. "
            "Adjust the call to match its signature.",
        ),
        # KeyError
        (
            r"KeyError: '(\w+)'",
            "The dictionary key '{0}' does not exist.\n"
            "Check available keys or use dict.get('{0}') with a default value.",
        ),
        # IndexError
        (
            r"IndexError: list index out of range",
            "A list index is out of range. Ensure the index is valid and"
            "within the length of the list.",
        ),
        # ValueError
        (
            r"ValueError: (.+)",
            "Invalid value: {0}. Double-check function arguments and data formats.",
        ),
        # FileNotFoundError
        (
            r"No such file or directory: '(.+)'",
            "The file '{0}' was not found. Check the path or create the file before accessing it.",
        ),
    ]

    @staticmethod
    def generate_correction_hints(error_text: str) -> list[str]:
        """Return a list of correction hints based on known error patterns."""
        hints = []
        for pattern, template in QueryErrorHandler.ERROR_HINTS:
            match = re.search(pattern, error_text)
            if match:
                hints.append(template.format(*match.groups()))
        if not hints:
            # Fallback hint
            hints.append(
                "An error occurred. Review the traceback carefully and ensure all variables, "
                "attributes, and imports are valid."
            )
        return hints

    @staticmethod
    def get_retry_instructions() -> str:
        """Return instructions for retry attempts."""
        return """
        Do *NOT* reintroduce any of these errors.

        Your task:
        - Modify the code so that it works end-to-end.
        - Preserve existing correct logic.
        - Avoid reintroducing any errors listed above.
        - Apply the correction hints above where relevant.
        - Reason about why the changes will not introduce new errors
        - Do *NOT* include reasoning in the updated explanation

        Before producing the code, do the following:
        1. List the specific lines that need to change from the previous attempt.
        2. Explain briefly what each change fixes.
        3. Then, provide the full corrected code.

        After writing the code, mentally simulate running it and check:
        - Does it still contain any of the errors from the error history?
        - Does it introduce any new undefined classes, methods, or attributes?
        If yes, fix them before finalizing.
        """

    @staticmethod
    def build_retry_query(
        execution_error: str,
        original_query: str,
        last_generated_code: str,
        error_history: list[str],
    ) -> str:
        """Build a retry query with appropriate error correction hints."""
        retry_query = (
            f"The previous result produced an error: {str(execution_error)}\n\n"
            f"The original prompt was: {original_query}\n\n"
            f"The code that failed was:\n\n```python\n{last_generated_code}\n```\n\n"
        )

        # Add error history if multiple failures
        if len(error_history) > 1:
            retry_query += "\nError History:\n"
            for i, err in enumerate(error_history):
                retry_query += f"{i + 1}. {err}\n"

        # Add correction hints
        hints = QueryErrorHandler.generate_correction_hints(str(execution_error))
        if hints:
            retry_query += "\nCorrection Hints:\n" + "\n".join(f"- {h}" for h in hints) + "\n"

        # Add retry instructions
        retry_query += QueryErrorHandler.get_retry_instructions()

        return retry_query
