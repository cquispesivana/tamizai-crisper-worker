FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    CRISPERWHISPER_CACHE=/models/crisperwhisper \
    CRISPER_MODEL_SIZE=medium \
    CRISPER_BACKEND=ct2 \
    CRISPER_DEVICE=cuda \
    CRISPER_COMPUTE_TYPE=float16 \
    CRISPER_LANGUAGE=es \
    CRISPER_MODE=verbatim \
    CRISPER_WORD_TIMESTAMPS=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
