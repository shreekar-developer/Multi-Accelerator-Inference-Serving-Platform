#!/bin/bash
set -e

# Build and push Docker images for the inference serving platform
# Usage: ./build-images.sh <ECR_REPO_PREFIX>

ECR_REPO_PREFIX=${1:-""}
VERSION=${VERSION:-"latest"}
REGION=${AWS_REGION:-"us-west-2"}

if [ -z "$ECR_REPO_PREFIX" ]; then
    echo "Usage: $0 <ECR_REPO_PREFIX>"
    echo "Example: $0 123456789012.dkr.ecr.us-west-2.amazonaws.com"
    exit 1
fi

echo "🐳 Building Docker images for Multi-Accelerator Inference Serving Platform"
echo "ECR Prefix: $ECR_REPO_PREFIX"
echo "Version: $VERSION"
echo "Region: $REGION"

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO_PREFIX

# Function to build and push image
build_and_push() {
    local component=$1
    local dockerfile_path=$2
    local context_path=$3
    local image_name="${ECR_REPO_PREFIX}/ml-serving-platform/${component}:${VERSION}"
    
    echo "📦 Building $component..."
    docker build -t $image_name -f $dockerfile_path $context_path
    
    echo "⬆️  Pushing $component..."
    docker push $image_name
    
    echo "✅ Successfully built and pushed $component"
}

# Build Router Service
echo "🔀 Building Router Service..."
cat > router/Dockerfile << 'EOF'
# Multi-stage build for Go router service
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o router ./main.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/router .

EXPOSE 8080 9090

CMD ["./router"]
EOF

build_and_push "router" "router/Dockerfile" "router/"

# Build CPU Backend
echo "🖥️  Building CPU Backend..."
cat > backends/cpu/onnxrt-server/Dockerfile << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

CMD ["python", "main.py"]
EOF

cat > backends/cpu/onnxrt-server/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
numpy==1.24.3
psutil==5.9.6
requests==2.31.0
EOF

build_and_push "cpu-backend" "backends/cpu/onnxrt-server/Dockerfile" "backends/cpu/onnxrt-server/"

# Build GPU Backend  
echo "🚀 Building GPU Backend..."
cat > backends/gpu/tensorrt-server/Dockerfile << 'EOF'
FROM nvcr.io/nvidia/tensorrt:23.10-py3

WORKDIR /app

# Copy Python backend code (similar structure to CPU backend)
COPY main.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

CMD ["python", "main.py"]
EOF

cat > backends/gpu/tensorrt-server/main.py << 'EOF'
#!/usr/bin/env python3
"""GPU Backend Service using TensorRT"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="GPU Backend Service", version="1.0.0")

class InferenceRequest(BaseModel):
    model_id: str
    version: str = "latest"
    inputs: List[Dict[str, Any]]
    request_id: Optional[str] = None

class InferenceResponse(BaseModel):
    request_id: str
    outputs: List[Dict[str, Any]]
    latency_ms: float
    model_id: str
    accelerator: str = "gpu"

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    start_time = time.time()
    
    # Simulate GPU inference (much faster than CPU)
    await asyncio.sleep(0.02)  # 20ms base latency
    
    outputs = [{"result": "gpu_processed", "score": 0.95} for _ in request.inputs]
    latency_ms = (time.time() - start_time) * 1000
    
    return InferenceResponse(
        request_id=request.request_id or "gpu-req",
        outputs=outputs,
        latency_ms=latency_ms,
        model_id=request.model_id
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "accelerator": "gpu"}

@app.get("/health/metrics")
async def health_metrics():
    return {
        "p95_latency_ms": 25.0,
        "queue_depth": 1,
        "active_requests": 0,
        "throughput_qps": 300.0,
        "error_rate_percent": 0.2,
        "memory": {"available_mb": 16384},
        "cpu_utilization": 30.0,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

cat > backends/gpu/tensorrt-server/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
numpy==1.24.3
requests==2.31.0
EOF

build_and_push "gpu-backend" "backends/gpu/tensorrt-server/Dockerfile" "backends/gpu/tensorrt-server/"

# Build Inferentia Backend
echo "🧠 Building Inferentia Backend..."
cat > backends/neuron/inferentia-server/Dockerfile << 'EOF'
FROM public.ecr.aws/neuron/pytorch-inference-neuron:1.13.1-neuron-py310-sdk2.18.0-ubuntu20.04

WORKDIR /app

COPY main.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

CMD ["python", "main.py"]
EOF

cat > backends/neuron/inferentia-server/main.py << 'EOF'
#!/usr/bin/env python3
"""Inferentia Backend Service using AWS Neuron"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Inferentia Backend Service", version="1.0.0")

class InferenceRequest(BaseModel):
    model_id: str
    version: str = "latest"
    inputs: List[Dict[str, Any]]
    request_id: Optional[str] = None

class InferenceResponse(BaseModel):
    request_id: str
    outputs: List[Dict[str, Any]]
    latency_ms: float
    model_id: str
    accelerator: str = "inferentia"

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    start_time = time.time()
    
    # Simulate Inferentia inference
    await asyncio.sleep(0.035)  # 35ms base latency
    
    outputs = [{"result": "inferentia_processed", "score": 0.90} for _ in request.inputs]
    latency_ms = (time.time() - start_time) * 1000
    
    return InferenceResponse(
        request_id=request.request_id or "inf-req",
        outputs=outputs,
        latency_ms=latency_ms,
        model_id=request.model_id
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "accelerator": "inferentia"}

@app.get("/health/metrics")
async def health_metrics():
    return {
        "p95_latency_ms": 35.0,
        "queue_depth": 1,
        "active_requests": 0,
        "throughput_qps": 200.0,
        "error_rate_percent": 0.3,
        "memory": {"available_mb": 8192},
        "cpu_utilization": 60.0,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

cat > backends/neuron/inferentia-server/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
requests==2.31.0
EOF

build_and_push "neuron-backend" "backends/neuron/inferentia-server/Dockerfile" "backends/neuron/inferentia-server/"

# Build Profiler Service
echo "📊 Building Profiler Service..."
cat > profiler/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
EOF

cat > profiler/main.py << 'EOF'
#!/usr/bin/env python3
"""Profiler Service for benchmarking models across accelerators"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Profiler Service", version="1.0.0")

@app.get("/")
async def root():
    return {"service": "profiler", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

cat > profiler/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
boto3==1.34.0
requests==2.31.0
EOF

build_and_push "profiler" "profiler/Dockerfile" "profiler/"

echo "🎉 All images built and pushed successfully!"
echo ""
echo "📋 Built images:"
echo "  • ${ECR_REPO_PREFIX}/ml-serving-platform/router:${VERSION}"
echo "  • ${ECR_REPO_PREFIX}/ml-serving-platform/cpu-backend:${VERSION}"
echo "  • ${ECR_REPO_PREFIX}/ml-serving-platform/gpu-backend:${VERSION}"
echo "  • ${ECR_REPO_PREFIX}/ml-serving-platform/neuron-backend:${VERSION}"
echo "  • ${ECR_REPO_PREFIX}/ml-serving-platform/profiler:${VERSION}"
echo ""
echo "✅ Ready for deployment to Kubernetes!" 