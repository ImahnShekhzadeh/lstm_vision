FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV MINICONDA_VERSION=Miniconda3-py310_24.3.0-0-Linux-x86_64.sh
ENV MINICONDA_SHA_256=\
43651393236cb8bb4219dcd429b3803a60f318e5507d8d84ca00dafa0c69f1bb

RUN apt-get update && apt-get install -y curl && \
    curl -O https://repo.anaconda.com/miniconda/$MINICONDA_VERSION && \
    /bin/bash $MINICONDA_VERSION -b -p /opt/conda && \
    rm $MINICONDA_VERSION && \
    apt-get -y install git && \
    apt-get clean

ENV PATH=/opt/conda/bin:$PATH
RUN conda init bash
RUN conda install -y python=3.10.3
RUN conda update -n base -c defaults conda
RUN conda install -c conda-forge pip

RUN conda install -y pytorch=2.1.* torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

WORKDIR /app

COPY setup.py .
COPY pyproject.toml .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 80

RUN git config --global --add safe.directory /app
