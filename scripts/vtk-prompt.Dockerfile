# vtk-prompt web UI / CLI image. Used by docker-compose.yml, which runs it next
# to a vtk-mcp server. From the repo root:
#
#   docker build -f scripts/vtk-prompt.Dockerfile -t vtk-prompt .

FROM python:3.12-slim

LABEL org.opencontainers.image.title="vtk-prompt"
LABEL org.opencontainers.image.source="https://github.com/Kitware/vtk-prompt"
LABEL org.opencontainers.image.licenses="MIT"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# No display in this container: force VTK's OSMesa software rasterizer, since
# the default GLX backend segfaults without an X server.
ENV VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

WORKDIR /app
COPY . /app/
# Editable: the prompt templates are not listed in the package data.
RUN pip install -e .

EXPOSE 8080
ENTRYPOINT ["vtk-prompt-ui"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--server"]
