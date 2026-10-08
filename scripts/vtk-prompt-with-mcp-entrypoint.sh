#!/bin/sh
# Runs vtk-prompt with --embed-mcp (which spawns vtk-mcp itself). Pass "ui" as
# the first argument to launch vtk-prompt-ui (also embedded) instead, e.g.
# `docker run -p 8080:8080 <image> ui`.
set -e

if [ "$1" = "ui" ]; then
    shift
    exec vtk-prompt-ui --embed-mcp --host 0.0.0.0 --server "$@"
fi

exec vtk-prompt --embed-mcp "$@"
