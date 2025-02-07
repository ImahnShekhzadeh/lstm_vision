FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git && \
    apt-get clean

COPY setup.py .
COPY pyproject.toml .

# Upgrade pip, install PyTorch and other packages via pip
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -e .

EXPOSE 80

RUN git config --global --add safe.directory /app
