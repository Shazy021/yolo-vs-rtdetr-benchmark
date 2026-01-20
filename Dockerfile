FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Alias python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install --no-cache-dir --upgrade pip
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py .
COPY visualize_metrics.py .
COPY config.yaml .
COPY src/ ./src/

RUN mkdir -p outputs metrics plots weights data

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
