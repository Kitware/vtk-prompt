# vtk-prompt-with-mcp: a single self-contained image bundling a local Qdrant
# server, vtk-mcp, and vtk-prompt (launched via its --embed-mcp flag). No
# docker compose, no separate vtk-mcp container, no manual startup step.
#
# The build context must be a directory containing both sibling repos as
# vtk-prompt/ and vtk-mcp/ (this is how CI checks them out; see
# .github/workflows/ci.yml's docker-deploy job). For local development,
# run from the parent directory of both checkouts:
#
#   docker build -f vtk-prompt/scripts/vtk-prompt-with-mcp.Dockerfile \
#                --ignorefile vtk-prompt/scripts/vtk-prompt-with-mcp.dockerignore \
#                -t vtk-prompt-with-mcp .
#   docker run --rm -it vtk-prompt-with-mcp "Create a red sphere" -t $ANTHROPIC_API_KEY
#
#   # Or the web UI instead of the CLI:
#   docker run --rm -it -p 8080:8080 vtk-prompt-with-mcp ui

FROM qdrant/qdrant:latest AS qdrant

FROM python:3.12-slim

LABEL org.opencontainers.image.title="vtk-prompt-with-mcp"
LABEL org.opencontainers.image.description="vtk-prompt with an embedded vtk-mcp server and local Qdrant"
LABEL org.opencontainers.image.licenses="MIT"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# VTK version to pre-cache at image build time
ARG VTK_VERSION=9.6.1
ENV VTK_MCP_VTK_VERSION=${VTK_VERSION}

# libunwind8 satisfies the Qdrant binary's runtime linker deps (it's built
# against glibc/libunwind on a fuller Debian base than python:3.12-slim).
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libunwind8 && \
    rm -rf /var/lib/apt/lists/*

# Bring in the Qdrant server binary + default config from the official image.
COPY --from=qdrant /qdrant/qdrant /qdrant/qdrant
COPY --from=qdrant /qdrant/config /qdrant/config

WORKDIR /app

RUN pip install uv

# Install vtk-* sibling packages from GitHub (not on PyPI). vtk-validate's
# "translate" extra (litellm) powers vtk-mcp's translate_prompt_to_dsl tool,
# which vtk-prompt calls by default (--dsl-translation).
RUN uv pip install --system \
    "git+https://github.com/vicentebolea/vtk-knowledge" \
    "vtk-validate[translate] @ git+https://github.com/vicentebolea/vtk-validate" \
    "git+https://github.com/vicentebolea/vtk-index"

COPY vtk-mcp/ /app/vtk-mcp/
COPY vtk-prompt/ /app/vtk-prompt/

RUN uv pip install --system -e "/app/vtk-mcp[retrieval]"
RUN uv pip install --system -e "/app/vtk-prompt"

# Pre-download the vtk-knowledge JSONL artifact and vtk-index embedded
# Qdrant storage so the image needs no network access to start serving.
RUN python /app/vtk-mcp/scripts/prefetch_artifacts.py

ENV VTK_MCP_ENABLE_VALIDATION=true

# No display in this container. VTK's default GLX/X11 render window still
# tries to open an X server even with OffScreenRenderingOn() (see
# rendering/scene_manager.py), which segfaults here. Force the OSMesa
# software-rasterizer backend that ships in the same "vtk" wheel instead, so
# both vtk-prompt-ui and any rendering the generated code does work headless.
ENV VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

COPY vtk-prompt/scripts/vtk-prompt-with-mcp-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh /qdrant/qdrant

# Qdrant is started for optional external indexing/inspection; see
# docker-entrypoint.sh for why vtk-mcp doesn't use it for retrieval.
EXPOSE 6333
# vtk-prompt-ui, when launched via `docker run <image> ui`.
EXPOSE 8080

ENTRYPOINT ["docker-entrypoint.sh"]
