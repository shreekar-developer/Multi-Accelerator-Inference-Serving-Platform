#!/usr/bin/env python3
"""
CPU Backend Service using ONNX Runtime for ARM64 (Graviton) processors
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional
import uuid
from dataclasses import dataclass, asdict
from collections import deque
import threading
import psutil

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CPU Backend Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    accelerator: str = "cpu"

@dataclass
class HealthMetrics:
    p95_latency_ms: float
    queue_depth: int
    active_requests: int
    throughput_qps: float
    error_rate_percent: float
    memory: Dict[str, int]
    cpu_utilization: float
    last_updated: str

class MetricsCollector:
    def __init__(self):
        self.latencies = deque(maxlen=1000)
        self.request_times = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        self.active_requests = 0
        self.queue_depth = 0
    
    def record_request(self, latency_ms: float, success: bool = True):
        self.latencies.append(latency_ms)
        self.request_times.append(time.time())
        self.total_requests += 1
        if not success:
            self.error_count += 1
    
    def get_p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(0.95 * len(sorted_latencies))
        return sorted_latencies[idx] if idx < len(sorted_latencies) else sorted_latencies[-1]
    
    def get_throughput_qps(self) -> float:
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t <= 60]
        return len(recent_requests) / 60.0
    
    def get_error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.error_count / self.total_requests) * 100.0

# Global metrics collector
metrics = MetricsCollector()

async def simulate_inference(model_id: str, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simulate model inference"""
    # Simulate processing time based on model
    if "bert" in model_id.lower():
        base_latency = 0.08
    elif "resnet" in model_id.lower():
        base_latency = 0.05
    else:
        base_latency = 0.03
    
    processing_time = base_latency * (1 + len(inputs) * 0.1)
    await asyncio.sleep(processing_time)
    
    outputs = []
    for _ in inputs:
        if "bert" in model_id.lower():
            outputs.append({"label": "POSITIVE", "score": 0.8 + np.random.random() * 0.2})
        elif "resnet" in model_id.lower():
            classes = ["Dog", "Cat", "Bird", "Car", "Person"]
            outputs.append({"class": np.random.choice(classes), "confidence": 0.7 + np.random.random() * 0.3})
        else:
            outputs.append({"result": "processed", "score": np.random.random()})
    
    return outputs

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Main inference endpoint"""
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    try:
        metrics.queue_depth += 1
        metrics.active_requests += 1
        
        outputs = await simulate_inference(request.model_id, request.inputs)
        latency_ms = (time.time() - start_time) * 1000
        
        metrics.record_request(latency_ms, success=True)
        
        return InferenceResponse(
            request_id=request_id,
            outputs=outputs,
            latency_ms=latency_ms,
            model_id=request.model_id
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request(latency_ms, success=False)
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        metrics.queue_depth -= 1
        metrics.active_requests -= 1

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/health/metrics")
async def health_metrics():
    """Health metrics for router"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    health_metrics = HealthMetrics(
        p95_latency_ms=metrics.get_p95_latency(),
        queue_depth=metrics.queue_depth,
        active_requests=metrics.active_requests,
        throughput_qps=metrics.get_throughput_qps(),
        error_rate_percent=metrics.get_error_rate(),
        memory={
            "used_mb": int(memory.used / 1024 / 1024),
            "available_mb": int(memory.available / 1024 / 1024),
        },
        cpu_utilization=cpu_percent,
        last_updated=time.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    
    return asdict(health_metrics)

@app.get("/models")
async def list_models():
    """List available models"""
    models = [
        {"model_id": "distilbert_sst2", "version": "v1.0"},
        {"model_id": "resnet50", "version": "v2.1"}
    ]
    return {"models": models, "count": len(models)}

@app.get("/")
async def root():
    return {
        "service": "CPU Backend Service",
        "accelerator": "cpu",
        "framework": "ONNX Runtime",
        "architecture": "ARM64",
        "status": "running"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting CPU Backend Service on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port, log_level="info") 