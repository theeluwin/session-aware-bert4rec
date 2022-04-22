# syntax=docker/dockerfile:experimental

# from
FROM theeluwin/pytorch-ko
LABEL maintainer="Jamie Seol <theeluwin@gmail.com>"

# apt
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git libgl1-mesa-dev libgtk2.0-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install packages
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
