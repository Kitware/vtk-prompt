"""Trame client module that loads the VTK/Python Monaco completion provider JS."""

from pathlib import Path

serve_path = str(Path(__file__).with_name("serve").resolve())

# Serve directory for JS files
serve = {"__vtk_completion": serve_path}

# Script loaded into the client; registers the Monaco completion provider.
scripts = ["__vtk_completion/vtk_completion.js"]
