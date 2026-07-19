FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements-deploy.txt ./requirements-deploy.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 \
    && pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user:user src ./src
COPY --chown=user:user web ./web
COPY --chown=user:user outputs/models/best_model_mel_v2.pth ./outputs/models/best_model_mel_v2.pth

USER user

EXPOSE 7860

CMD ["uvicorn", "web.backend:app", "--host", "0.0.0.0", "--port", "7860"]
