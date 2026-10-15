#!/bin/sh
# Starts the web UI on port 8080. With VTK_PROMPT_AUTH=true it starts behind the
# Google sign-in gateway instead (one UI process and one data folder per user).
set -e

if [ "$VTK_PROMPT_AUTH" = "true" ]; then
    exec vtk-prompt-gateway --host 0.0.0.0 --port 8080 "$@"
fi

exec vtk-prompt-ui --host 0.0.0.0 --port 8080 --server "$@"
