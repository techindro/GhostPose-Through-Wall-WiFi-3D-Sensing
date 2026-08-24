# GPU-Accelerated RF-Pose3D Inference Container
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY analytics/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY analytics /app/analytics

WORKDIR /app/analytics

EXPOSE 8000

CMD ["uvicorn", "kafka_inference_service:app", "--host", "0.0.0.0", "--port", "8000"]
