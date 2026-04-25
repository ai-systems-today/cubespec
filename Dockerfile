# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────────
# cubespec Python image — installs the package + JupyterLab.
#   default CMD: drop into the cubespec CLI help
#   override with `jupyter lab --ip 0.0.0.0` for notebook access
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . /work/python
RUN pip install -e "/work/python[plot,cli,calibration]" jupyterlab>=4

EXPOSE 8888
CMD ["cubespec", "--help"]
