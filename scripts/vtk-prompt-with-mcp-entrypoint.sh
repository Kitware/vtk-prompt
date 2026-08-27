#!/bin/sh
# Starts the local Qdrant process (available on :6333 for anyone who wants to
# index or inspect it), waits for it to accept connections, then runs
# vtk-prompt with --embed-mcp (which in turn spawns vtk-mcp itself). Pass
# "ui" as the first argument to launch vtk-prompt-ui (also embedded) instead,
# e.g. `docker run -p 8080:8080 <image> ui`.
#
# vtk-mcp is intentionally NOT pointed at this Qdrant instance: it starts
# empty, and the on-disk format vtk-index's embedded (client-local-mode)
# storage writes is not directly loadable by the full Qdrant server's
# raft-tracked collection registry, so retrieval would silently return
# nothing. vtk-mcp instead uses its own prefetched embedded storage
# (baked into the image at build time), which is the supported, working
# path. The Qdrant process here is for optional external use.
set -e

/qdrant/qdrant &
QDRANT_PID=$!
trap 'kill "$QDRANT_PID" 2>/dev/null' EXIT

echo "Waiting for Qdrant to become ready..." >&2
python - <<'PYEOF'
import time
import urllib.request

for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:6333/readyz", timeout=1)
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("Qdrant did not become ready in time")
PYEOF

if [ "$1" = "ui" ]; then
    shift
    exec vtk-prompt-ui --embed-mcp --host 0.0.0.0 --server "$@"
fi

exec vtk-prompt --embed-mcp "$@"
