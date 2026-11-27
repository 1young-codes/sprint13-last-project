# GPU 생성용 이미지. CUDA 런타임은 torch 휠에 들어 있어서 python 기본 이미지로 충분하다.
# 모델(/data/models), 결과(/data/outputs), 데이터(/data/datasets)는 이미지에 넣지 않고 마운트한다.
FROM python:3.12-slim AS gpu

# triton이 실행 중에 작은 C 커널을 컴파일하므로 gcc가 필요하다
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY . .

ENV HF_HOME=/data/models/hf_cache \
    PYTHONUNBUFFERED=1

FROM gpu AS ad-layers
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright
RUN pip install --no-cache-dir -r models/generative_ai/ad_layers/requirements.txt \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

FROM gpu AS runtime
