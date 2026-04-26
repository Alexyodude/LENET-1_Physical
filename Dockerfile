FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install . --no-deps \
 && pip install \
      "torch>=2.2" "torchvision>=0.17" "numpy>=1.26" "pyyaml>=6.0" \
      "fastapi>=0.111" "uvicorn[standard]>=0.30" "websockets>=12" "pydantic>=2.7"

COPY src ./src
COPY config ./config
COPY weights ./weights
COPY README.md ./

EXPOSE 7860

CMD ["python", "-m", "lenet1_physical.main", \
     "--mapping", "config/mapping.example.yaml", \
     "--weights", "weights/lenet5.pt", \
     "--mode", "simulate", \
     "--demo", \
     "--host", "0.0.0.0", \
     "--port", "7860"]
