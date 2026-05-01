FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# RUN apt-get update && apt-get install -y curl git \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt