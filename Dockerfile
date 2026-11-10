# CPU image. weights: put the .pth here or pass --build-arg WEIGHTS_URL=...

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# opencv / mediapipe libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# cpu torch, much smaller image
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# code, threshold, weights if present
COPY main.py schemas.py batcher.py tracking.py model.py gradcam.py index.html fine_tuned_medical_resnet.pt[h] ./
COPY results/metrics.json results/metrics.json

ARG WEIGHTS_URL=""
RUN if [ ! -f fine_tuned_medical_resnet.pth ] && [ -n "$WEIGHTS_URL" ]; then \
        python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], 'fine_tuned_medical_resnet.pth')" "$WEIGHTS_URL"; \
    fi \
    && if [ ! -f fine_tuned_medical_resnet.pth ]; then \
        echo 'WARNING: no fine-tuned weights in image; predictions will not be meaningful.'; \
    fi

RUN useradd --create-home --uid 1000 engine && chown -R engine /app
USER engine

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
