"""Run the VTK Prompt UI against trame's react client."""

from typing import Any

from trame.decorators import controller

from .controllers import conversation
from .vtk_prompt_ui import VTKPromptApp, run

try:
    from .ui_react import build_ui
except ImportError as exc:  # the react widgets ship as an extra
    raise SystemExit(
        "The react client needs extra packages: pip install 'vtk-prompt[react]'"
    ) from exc


class VTKPromptReactApp(VTKPromptApp):
    """VTK Prompt running against trame's react client."""

    CLIENT_TYPE = "react"

    def _build_ui(self) -> None:
        build_ui(self)

    @controller.set("import_uploaded_files")
    def import_uploaded_files(self, files: list[dict[str, Any]]) -> None:
        """Handle files read client-side by utils.js.

        The react client has no widget that serializes File objects into state,
        so the browser reads them and posts the same shape vuetify's VFileInput
        produced.
        """
        conversation.process_uploaded_files(self, files)
        self.state.import_dialog = False


def main() -> None:
    """Start the trame app with the react client."""
    run(VTKPromptReactApp)


if __name__ == "__main__":
    main()
