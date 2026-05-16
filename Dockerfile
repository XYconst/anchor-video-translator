# Dockerfile for the video-translator-vb FastAPI backend, built from the
# apps/video-translator-vb subdir so the pipeline can also access the
# Rubik font under frontend/public/fonts.
#
# Pipeline needs:
# - ffmpeg (audio/video extraction, encoding)
# - torch + demucs (vocal isolation, best-effort)
# - playwright + chromium (background replacement screenshots)
# - opencv (matting)
# We install torch from the CPU wheel index to keep the runtime image small;
# the CUDA wheels are 4-5 GB each and useless in a CPU container.

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps:
# - ffmpeg + libsndfile1 for audio/video
# - chromium runtime libs for playwright (the standard list — nss, gtk, etc.)
# - libgl + libglib for opencv
# - build-essential + git for pip wheels that fall back to source
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      libgl1 \
      libglib2.0-0 \
      ca-certificates \
      curl \
      tini \
      build-essential \
      git \
      libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
      libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
      libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt

# Install torch CPU-only first so resolver doesn't pull the 4 GB CUDA wheels.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r /app/backend/requirements.txt \
 && python -m playwright install chromium

# Source last so code changes don't bust the dep layer. Copy both backend code
# and the frontend fonts dir (the pipeline reads Rubik-Bold.ttf for the
# subtitle burn-in step).
COPY backend /app/backend
COPY frontend/public/fonts /app/frontend/public/fonts

WORKDIR /app/backend

# Volume mount for uploads / job working dirs — Railway provides this.
RUN mkdir -p /data/uploads
ENV UPLOAD_DIR=/data/uploads

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
