FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y curl ca-certificates git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY setup.py .
COPY pyproject.toml .

# Install uv acc. to the instructions https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
ADD https://astral.sh/uv/0.5.29/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install python with uv
RUN uv python install 3.11.11 && \
    uv venv --python 3.11.11 && \
    uv pip install -e .

RUN git config --global --add safe.directory /app