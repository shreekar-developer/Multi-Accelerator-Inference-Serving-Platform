#!/usr/bin/env python3
"""
Comprehensive Model Benchmarking Harness
Profiles models across different accelerators with detailed performance analysis
"""

import argparse
import json
import logging
import os
import statistics
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import psutil
import boto3
from botocore.exceptions import ClientError

# Import model-specific benchmarking modules
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    model_path: str
    model_name: str
    accelerator: str
    framework: str
    input_shape: Tuple[int, ...]
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    num_iterations: int = 100
    warmup_iterations: int = 10
    concurrent_requests: List[int] = None
    precision_modes: List[str] = None
    test_duration_seconds: int = 60
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 256, 512]
        if self.concurrent_requests is None:
            self.concurrent_requests = [1, 4, 8, 16, 32]
        if self.precision_modes is None:
            self.precision_modes = ["fp32", "fp16"]

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    model_name: str
    accelerator: str
    framework: str
    batch_size: int
    sequence_length: Optional[int]
    concurrent_requests: int
    precision: str
    
    # Latency metrics (in milliseconds)
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    
    # Throughput metrics
    throughput_qps: float
    requests_per_second: float
    
    # Resource utilization
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_memory_mb: float
    max_memory_mb: float
    avg_gpu_percent: Optional[float] = None
    max_gpu_percent: Optional[float] = None
    avg_gpu_memory_mb: Optional[float] = None
    max_gpu_memory_mb: Optional[float] = None
    
    # Model metrics
    model_size_mb: float
    parameters_count: Optional[int] = None
    
    # Test configuration
    num_iterations: int = 0
    test_duration_seconds: float = 0.0
    warmup_iterations: int = 0
    
    # Cold start metrics
    cold_start_latency_ms: Optional[float] = None
    first_request_latency_ms: Optional[float] = None
    
    # Error metrics
    error_count: int = 0
    error_rate_percent: float = 0.0
    
    # Additional metadata
    timestamp: float = 0.0
    environment_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.environment_info is None:
            self.environment_info = {}

class ModelBenchmarker:
    """Handles benchmarking of models across different accelerators"""
    
    def __init__(
        self,
        output_dir: str = "benchmarks/results",
        s3_bucket: Optional[str] = None,
        dynamodb_table: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.s3_bucket = s3_bucket
        self.dynamodb_table = dynamodb_table
        
        # Initialize AWS clients if configured
        self.s3_client = boto3.client('s3') if s3_bucket else None
        self.dynamodb_client = boto3.client('dynamodb') if dynamodb_table else None
        
        # Initialize monitoring
        self.system_monitor = SystemMonitor()
        
        # GPU monitoring if available
        self.gpu_monitor = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_monitor = GPUMonitor()
            logger.info("GPU monitoring enabled")
        except ImportError:
            logger.info("GPU monitoring not available (pynvml not installed)")
    
    def benchmark_model(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Benchmark a model with the given configuration"""
        logger.info(f"Starting benchmark for {config.model_name} on {config.accelerator}")
        
        results = []
        
        # Load model based on framework
        model_loader = self._get_model_loader(config)
        if not model_loader:
            logger.error(f"No model loader available for {config.framework}")
            return results
        
        try:
            model = model_loader.load_model(config.model_path)
            model_info = model_loader.get_model_info(model)
            
            # Run benchmarks for different configurations
            for batch_size in config.batch_sizes:
                for seq_len in config.sequence_lengths if config.sequence_lengths else [None]:
                    for concurrent in config.concurrent_requests:
                        for precision in config.precision_modes:
                            
                            if not model_loader.supports_precision(precision):
                                logger.info(f"Skipping {precision} (not supported)")
                                continue
                            
                            logger.info(f"Benchmarking: batch={batch_size}, seq={seq_len}, "
                                      f"concurrent={concurrent}, precision={precision}")
                            
                            try:
                                result = self._run_single_benchmark(
                                    model, model_loader, config, batch_size, seq_len, 
                                    concurrent, precision, model_info
                                )
                                results.append(result)
                                
                            except Exception as e:
                                logger.error(f"Benchmark failed for configuration: {str(e)}")
                                logger.error(traceback.format_exc())
                                
                                # Create error result
                                error_result = BenchmarkResult(
                                    model_name=config.model_name,
                                    accelerator=config.accelerator,
                                    framework=config.framework,
                                    batch_size=batch_size,
                                    sequence_length=seq_len,
                                    concurrent_requests=concurrent,
                                    precision=precision,
                                    mean_latency_ms=0.0,
                                    p50_latency_ms=0.0,
                                    p95_latency_ms=0.0,
                                    p99_latency_ms=0.0,
                                    min_latency_ms=0.0,
                                    max_latency_ms=0.0,
                                    std_latency_ms=0.0,
                                    throughput_qps=0.0,
                                    requests_per_second=0.0,
                                    avg_cpu_percent=0.0,
                                    max_cpu_percent=0.0,
                                    avg_memory_mb=0.0,
                                    max_memory_mb=0.0,
                                    model_size_mb=model_info.get('size_mb', 0.0),
                                    error_count=1,
                                    error_rate_percent=100.0,
                                    environment_info={'error': str(e)}
                                )
                                results.append(error_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to load model {config.model_path}: {str(e)}")
            return []
    
    def _run_single_benchmark(
        self,
        model,
        model_loader,
        config: BenchmarkConfig,
        batch_size: int,
        sequence_length: Optional[int],
        concurrent_requests: int,
        precision: str,
        model_info: Dict[str, Any]
    ) -> BenchmarkResult:
        """Run a single benchmark configuration"""
        
        # Prepare inputs
        inputs = self._generate_inputs(
            config.input_shape, batch_size, sequence_length, config.framework
        )
        
        # Set precision if supported
        if hasattr(model_loader, 'set_precision'):
            model_loader.set_precision(model, precision)
        
        # Start monitoring
        self.system_monitor.start_monitoring()
        if self.gpu_monitor:
            self.gpu_monitor.start_monitoring()
        
        # Warmup
        logger.debug(f"Warming up with {config.warmup_iterations} iterations...")
        for _ in range(config.warmup_iterations):
            try:
                _ = model_loader.run_inference(model, inputs)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {str(e)}")
        
        # Cold start measurement
        cold_start_time = time.time()
        try:
            first_output = model_loader.run_inference(model, inputs)
            cold_start_latency = (time.time() - cold_start_time) * 1000
        except Exception as e:
            logger.warning(f"Cold start measurement failed: {str(e)}")
            cold_start_latency = None
        
        # Main benchmark
        latencies = []
        error_count = 0
        test_start_time = time.time()
        
        if concurrent_requests == 1:
            # Sequential execution
            latencies, error_count = self._run_sequential_benchmark(
                model, model_loader, inputs, config.num_iterations
            )
        else:
            # Concurrent execution
            latencies, error_count = self._run_concurrent_benchmark(
                model, model_loader, inputs, concurrent_requests, config.test_duration_seconds
            )
        
        test_duration = time.time() - test_start_time
        
        # Stop monitoring
        cpu_stats = self.system_monitor.stop_monitoring()
        gpu_stats = self.gpu_monitor.stop_monitoring() if self.gpu_monitor else {}
        
        # Calculate statistics
        if latencies:
            latency_stats = self._calculate_latency_statistics(latencies)
            throughput_qps = len(latencies) / test_duration
            requests_per_second = len(latencies) / test_duration
        else:
            latency_stats = {
                'mean': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0,
                'min': 0.0, 'max': 0.0, 'std': 0.0
            }
            throughput_qps = 0.0
            requests_per_second = 0.0
        
        # Create result
        result = BenchmarkResult(
            model_name=config.model_name,
            accelerator=config.accelerator,
            framework=config.framework,
            batch_size=batch_size,
            sequence_length=sequence_length,
            concurrent_requests=concurrent_requests,
            precision=precision,
            
            # Latency metrics
            mean_latency_ms=latency_stats['mean'],
            p50_latency_ms=latency_stats['p50'],
            p95_latency_ms=latency_stats['p95'],
            p99_latency_ms=latency_stats['p99'],
            min_latency_ms=latency_stats['min'],
            max_latency_ms=latency_stats['max'],
            std_latency_ms=latency_stats['std'],
            
            # Throughput metrics
            throughput_qps=throughput_qps,
            requests_per_second=requests_per_second,
            
            # Resource utilization
            avg_cpu_percent=cpu_stats.get('avg_cpu_percent', 0.0),
            max_cpu_percent=cpu_stats.get('max_cpu_percent', 0.0),
            avg_memory_mb=cpu_stats.get('avg_memory_mb', 0.0),
            max_memory_mb=cpu_stats.get('max_memory_mb', 0.0),
            avg_gpu_percent=gpu_stats.get('avg_gpu_percent'),
            max_gpu_percent=gpu_stats.get('max_gpu_percent'),
            avg_gpu_memory_mb=gpu_stats.get('avg_gpu_memory_mb'),
            max_gpu_memory_mb=gpu_stats.get('max_gpu_memory_mb'),
            
            # Model metrics
            model_size_mb=model_info.get('size_mb', 0.0),
            parameters_count=model_info.get('parameters_count'),
            
            # Test configuration
            num_iterations=len(latencies),
            test_duration_seconds=test_duration,
            warmup_iterations=config.warmup_iterations,
            
            # Cold start metrics
            cold_start_latency_ms=cold_start_latency,
            first_request_latency_ms=latencies[0] if latencies else None,
            
            # Error metrics
            error_count=error_count,
            error_rate_percent=(error_count / max(len(latencies) + error_count, 1)) * 100,
            
            # Environment info
            environment_info=self._get_environment_info(config)
        )
        
        return result
    
    def _run_sequential_benchmark(self, model, model_loader, inputs, num_iterations: int) -> Tuple[List[float], int]:
        """Run sequential benchmark"""
        latencies = []
        error_count = 0
        
        for i in range(num_iterations):
            start_time = time.time()
            try:
                _ = model_loader.run_inference(model, inputs)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            except Exception as e:
                error_count += 1
                logger.debug(f"Inference error in iteration {i}: {str(e)}")
        
        return latencies, error_count
    
    def _run_concurrent_benchmark(
        self, 
        model, 
        model_loader, 
        inputs, 
        concurrent_requests: int, 
        duration_seconds: int
    ) -> Tuple[List[float], int]:
        """Run concurrent benchmark"""
        latencies = []
        error_count = 0
        start_time = time.time()
        
        def run_single_request():
            request_start = time.time()
            try:
                _ = model_loader.run_inference(model, inputs)
                return (time.time() - request_start) * 1000
            except Exception as e:
                logger.debug(f"Inference error: {str(e)}")
                return None
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = []
            
            # Submit initial batch of requests
            for _ in range(concurrent_requests):
                future = executor.submit(run_single_request)
                futures.append(future)
            
            # Continue submitting requests until duration is reached
            while (time.time() - start_time) < duration_seconds:
                # Check for completed futures
                completed_futures = []
                for future in futures:
                    if future.done():
                        completed_futures.append(future)
                        
                        try:
                            result = future.result()
                            if result is not None:
                                latencies.append(result)
                            else:
                                error_count += 1
                        except Exception as e:
                            error_count += 1
                            logger.debug(f"Future error: {str(e)}")
                
                # Remove completed futures and submit new ones
                for future in completed_futures:
                    futures.remove(future)
                    if (time.time() - start_time) < duration_seconds:
                        new_future = executor.submit(run_single_request)
                        futures.append(new_future)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
            
            # Wait for remaining futures to complete
            for future in as_completed(futures, timeout=30):
                try:
                    result = future.result()
                    if result is not None:
                        latencies.append(result)
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Final future error: {str(e)}")
        
        return latencies, error_count
    
    def _calculate_latency_statistics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate latency statistics"""
        if not latencies:
            return {'mean': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        
        latencies_array = np.array(latencies)
        
        return {
            'mean': float(np.mean(latencies_array)),
            'p50': float(np.percentile(latencies_array, 50)),
            'p95': float(np.percentile(latencies_array, 95)),
            'p99': float(np.percentile(latencies_array, 99)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array)),
            'std': float(np.std(latencies_array))
        }
    
    def _generate_inputs(
        self, 
        input_shape: Tuple[int, ...], 
        batch_size: int, 
        sequence_length: Optional[int],
        framework: str
    ):
        """Generate appropriate inputs for the model"""
        shape = list(input_shape)
        shape[0] = batch_size  # Set batch size
        
        if sequence_length is not None and len(shape) > 1:
            shape[1] = sequence_length  # Set sequence length for NLP models
        
        if framework == "onnx":
            return np.random.randn(*shape).astype(np.float32)
        elif framework == "pytorch":
            return torch.randn(*shape)
        elif framework == "tensorrt":
            return np.random.randn(*shape).astype(np.float32)
        else:
            return np.random.randn(*shape).astype(np.float32)
    
    def _get_model_loader(self, config: BenchmarkConfig):
        """Get appropriate model loader based on framework"""
        if config.framework == "onnx" and ONNX_AVAILABLE:
            return ONNXModelLoader()
        elif config.framework == "pytorch" and PYTORCH_AVAILABLE:
            return PyTorchModelLoader()
        elif config.framework == "tensorrt" and TENSORRT_AVAILABLE:
            return TensorRTModelLoader()
        else:
            logger.error(f"Model loader not available for framework: {config.framework}")
            return None
    
    def _get_environment_info(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Get environment information"""
        import platform
        
        env_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'framework': config.framework,
            'accelerator': config.accelerator
        }
        
        # Add GPU info if available
        if self.gpu_monitor:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                env_info['gpu_name'] = gpu_name
                env_info['gpu_memory_total_gb'] = gpu_memory.total / (1024**3)
            except Exception:
                pass
        
        return env_info
    
    def save_results(self, results: List[BenchmarkResult], output_file: Optional[str] = None) -> str:
        """Save benchmark results to file and optionally to S3/DynamoDB"""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"benchmark_results_{timestamp}.json"
        
        output_path = self.output_dir / output_file
        
        # Convert to JSON-serializable format
        results_data = [asdict(result) for result in results]
        
        # Save to local file
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        
        # Upload to S3 if configured
        if self.s3_client and self.s3_bucket:
            try:
                s3_key = f"benchmark-results/{output_file}"
                self.s3_client.upload_file(str(output_path), self.s3_bucket, s3_key)
                logger.info(f"Results uploaded to s3://{self.s3_bucket}/{s3_key}")
            except ClientError as e:
                logger.error(f"Failed to upload to S3: {str(e)}")
        
        # Store in DynamoDB if configured
        if self.dynamodb_client and self.dynamodb_table:
            try:
                self._store_results_in_dynamodb(results)
                logger.info("Results stored in DynamoDB")
            except Exception as e:
                logger.error(f"Failed to store in DynamoDB: {str(e)}")
        
        return str(output_path)
    
    def _store_results_in_dynamodb(self, results: List[BenchmarkResult]):
        """Store results in DynamoDB"""
        for result in results:
            item = {
                'benchmark_id': {'S': f"{result.model_name}#{result.accelerator}#{int(result.timestamp)}"},
                'timestamp': {'N': str(int(result.timestamp))},
                'model_name': {'S': result.model_name},
                'accelerator': {'S': result.accelerator},
                'framework': {'S': result.framework},
                'batch_size': {'N': str(result.batch_size)},
                'concurrent_requests': {'N': str(result.concurrent_requests)},
                'precision': {'S': result.precision},
                'p95_latency_ms': {'N': str(result.p95_latency_ms)},
                'throughput_qps': {'N': str(result.throughput_qps)},
                'error_rate_percent': {'N': str(result.error_rate_percent)},
            }
            
            if result.sequence_length is not None:
                item['sequence_length'] = {'N': str(result.sequence_length)}
            
            self.dynamodb_client.put_item(
                TableName=self.dynamodb_table,
                Item=item
            )

class SystemMonitor:
    """Monitors system resource usage during benchmarking"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_percentages = []
        self.memory_usages = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.cpu_percentages = []
        self.memory_usages = []
        
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if self.cpu_percentages and self.memory_usages:
            return {
                'avg_cpu_percent': statistics.mean(self.cpu_percentages),
                'max_cpu_percent': max(self.cpu_percentages),
                'avg_memory_mb': statistics.mean(self.memory_usages),
                'max_memory_mb': max(self.memory_usages)
            }
        else:
            return {
                'avg_cpu_percent': 0.0,
                'max_cpu_percent': 0.0,
                'avg_memory_mb': 0.0,
                'max_memory_mb': 0.0
            }
    
    def _monitor_loop(self):
        """Monitor loop running in separate thread"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_mb = psutil.virtual_memory().used / (1024**2)
                
                self.cpu_percentages.append(cpu_percent)
                self.memory_usages.append(memory_mb)
            except Exception as e:
                logger.debug(f"Monitoring error: {str(e)}")
            
            time.sleep(0.1)

class GPUMonitor:
    """Monitors GPU resource usage during benchmarking"""
    
    def __init__(self):
        self.monitoring = False
        self.gpu_percentages = []
        self.gpu_memory_usages = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start monitoring GPU resources"""
        try:
            import pynvml
            self.monitoring = True
            self.gpu_percentages = []
            self.gpu_memory_usages = []
            
            import threading
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.start()
        except ImportError:
            logger.warning("GPU monitoring not available")
    
    def stop_monitoring(self) -> Dict[str, Optional[float]]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if self.gpu_percentages and self.gpu_memory_usages:
            return {
                'avg_gpu_percent': statistics.mean(self.gpu_percentages),
                'max_gpu_percent': max(self.gpu_percentages),
                'avg_gpu_memory_mb': statistics.mean(self.gpu_memory_usages),
                'max_gpu_memory_mb': max(self.gpu_memory_usages)
            }
        else:
            return {
                'avg_gpu_percent': None,
                'max_gpu_percent': None,
                'avg_gpu_memory_mb': None,
                'max_gpu_memory_mb': None
            }
    
    def _monitor_loop(self):
        """Monitor loop running in separate thread"""
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            while self.monitoring:
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    self.gpu_percentages.append(utilization.gpu)
                    self.gpu_memory_usages.append(memory_info.used / (1024**2))
                except Exception as e:
                    logger.debug(f"GPU monitoring error: {str(e)}")
                
                time.sleep(0.1)
        except ImportError:
            pass

# Model loader implementations
class ONNXModelLoader:
    """ONNX model loader and runner"""
    
    def __init__(self):
        self.session = None
    
    def load_model(self, model_path: str):
        """Load ONNX model"""
        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        return self.session
    
    def run_inference(self, model, inputs):
        """Run inference with ONNX model"""
        input_name = model.get_inputs()[0].name
        return model.run(None, {input_name: inputs})
    
    def get_model_info(self, model) -> Dict[str, Any]:
        """Get model information"""
        return {
            'size_mb': 0.0,  # Would need to calculate from file size
            'input_shape': [inp.shape for inp in model.get_inputs()],
            'output_shape': [out.shape for out in model.get_outputs()]
        }
    
    def supports_precision(self, precision: str) -> bool:
        """Check if precision is supported"""
        return precision in ['fp32', 'fp16']

class PyTorchModelLoader:
    """PyTorch model loader and runner"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_path: str):
        """Load PyTorch model"""
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model
    
    def run_inference(self, model, inputs):
        """Run inference with PyTorch model"""
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(self.device)
        else:
            inputs = inputs.to(self.device)
        
        with torch.no_grad():
            return model(inputs)
    
    def get_model_info(self, model) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in model.parameters())
        return {
            'size_mb': 0.0,  # Would need to calculate from model size
            'parameters_count': total_params
        }
    
    def supports_precision(self, precision: str) -> bool:
        """Check if precision is supported"""
        return precision in ['fp32', 'fp16']
    
    def set_precision(self, model, precision: str):
        """Set model precision"""
        if precision == 'fp16':
            model.half()
        elif precision == 'fp32':
            model.float()

class TensorRTModelLoader:
    """TensorRT model loader and runner"""
    
    def __init__(self):
        self.runtime = None
        self.engine = None
        self.context = None
        self.logger = trt.Logger(trt.Logger.WARNING)
    
    def load_model(self, model_path: str):
        """Load TensorRT engine"""
        with open(model_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        return self.engine
    
    def run_inference(self, model, inputs):
        """Run inference with TensorRT engine"""
        # This is a simplified implementation
        # Real implementation would handle GPU memory management
        bindings = []
        
        # Allocate device memory for inputs and outputs
        for i in range(model.num_bindings):
            binding_shape = model.get_binding_shape(i)
            binding_dtype = trt.nptype(model.get_binding_dtype(i))
            
            size = trt.volume(binding_shape) * model.max_batch_size
            device_mem = cuda.mem_alloc(size * binding_dtype().itemsize)
            bindings.append(int(device_mem))
            
            if model.binding_is_input(i):
                cuda.memcpy_htod(device_mem, inputs.astype(binding_dtype))
        
        # Run inference
        self.context.execute_v2(bindings)
        
        return "inference_complete"  # Simplified return
    
    def get_model_info(self, model) -> Dict[str, Any]:
        """Get model information"""
        return {
            'size_mb': 0.0,  # Would calculate from engine size
            'max_batch_size': model.max_batch_size,
            'num_bindings': model.num_bindings
        }
    
    def supports_precision(self, precision: str) -> bool:
        """Check if precision is supported"""
        return precision in ['fp32', 'fp16', 'int8']

def create_benchmark_config_from_model_files(
    model_dir: str,
    accelerator: str = "cpu",
    framework: str = "onnx"
) -> List[BenchmarkConfig]:
    """Create benchmark configurations from model files in directory"""
    model_dir = Path(model_dir)
    configs = []
    
    extensions = {
        'onnx': ['.onnx'],
        'pytorch': ['.pt', '.pth'],
        'tensorrt': ['.trt', '.engine']
    }
    
    for model_file in model_dir.rglob("*"):
        if model_file.suffix in extensions.get(framework, []):
            config = BenchmarkConfig(
                model_path=str(model_file),
                model_name=model_file.stem,
                accelerator=accelerator,
                framework=framework,
                input_shape=(1, 3, 224, 224)  # Default vision model shape
            )
            configs.append(config)
    
    return configs

def main():
    parser = argparse.ArgumentParser(description='Comprehensive model benchmarking')
    parser.add_argument('--model-path', help='Path to model file')
    parser.add_argument('--model-dir', help='Directory containing models to benchmark')
    parser.add_argument('--model-name', help='Model name')
    parser.add_argument('--accelerator', choices=['cpu', 'gpu', 'inferentia'], default='cpu')
    parser.add_argument('--framework', choices=['onnx', 'pytorch', 'tensorrt'], default='onnx')
    parser.add_argument('--input-shape', help='Input shape (comma-separated)', default='1,3,224,224')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument('--sequence-lengths', nargs='+', type=int, help='For NLP models')
    parser.add_argument('--concurrent-requests', nargs='+', type=int, default=[1, 4, 8, 16])
    parser.add_argument('--precision-modes', nargs='+', default=['fp32', 'fp16'])
    parser.add_argument('--num-iterations', type=int, default=100)
    parser.add_argument('--warmup-iterations', type=int, default=10)
    parser.add_argument('--test-duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--output-dir', default='benchmarks/results')
    parser.add_argument('--output-file', help='Output filename')
    parser.add_argument('--s3-bucket', help='S3 bucket for results upload')
    parser.add_argument('--dynamodb-table', help='DynamoDB table for results storage')
    parser.add_argument('--parallel-models', action='store_true', help='Benchmark models in parallel')
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker(
        output_dir=args.output_dir,
        s3_bucket=args.s3_bucket,
        dynamodb_table=args.dynamodb_table
    )
    
    # Create benchmark configurations
    configs = []
    
    if args.model_path and args.model_name:
        # Single model benchmark
        config = BenchmarkConfig(
            model_path=args.model_path,
            model_name=args.model_name,
            accelerator=args.accelerator,
            framework=args.framework,
            input_shape=tuple(map(int, args.input_shape.split(','))),
            batch_sizes=args.batch_sizes,
            sequence_lengths=args.sequence_lengths,
            concurrent_requests=args.concurrent_requests,
            precision_modes=args.precision_modes,
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations,
            test_duration_seconds=args.test_duration
        )
        configs.append(config)
    
    elif args.model_dir:
        # Multiple models benchmark
        configs = create_benchmark_config_from_model_files(
            args.model_dir, args.accelerator, args.framework
        )
        
        # Update configs with command line arguments
        for config in configs:
            config.batch_sizes = args.batch_sizes
            config.sequence_lengths = args.sequence_lengths
            config.concurrent_requests = args.concurrent_requests
            config.precision_modes = args.precision_modes
            config.num_iterations = args.num_iterations
            config.warmup_iterations = args.warmup_iterations
            config.test_duration_seconds = args.test_duration
    
    else:
        parser.print_help()
        return
    
    # Run benchmarks
    all_results = []
    
    if args.parallel_models and len(configs) > 1:
        # Benchmark models in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_config = {
                executor.submit(benchmarker.benchmark_model, config): config 
                for config in configs
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"Completed benchmark for {config.model_name}")
                except Exception as e:
                    logger.error(f"Benchmark failed for {config.model_name}: {str(e)}")
    else:
        # Sequential benchmarking
        for config in configs:
            logger.info(f"Benchmarking {config.model_name}...")
            results = benchmarker.benchmark_model(config)
            all_results.extend(results)
    
    # Save results
    if all_results:
        output_file = benchmarker.save_results(all_results, args.output_file)
        
        # Print summary
        successful_results = [r for r in all_results if r.error_count == 0]
        failed_results = [r for r in all_results if r.error_count > 0]
        
        print(f"\n=== Benchmark Summary ===")
        print(f"Total benchmark runs: {len(all_results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        print(f"Results saved to: {output_file}")
        
        if successful_results:
            print(f"\n=== Performance Summary ===")
            for result in successful_results[:10]:  # Show top 10 results
                print(f"{result.model_name} ({result.accelerator}, {result.framework}, "
                      f"batch={result.batch_size}): "
                      f"p95={result.p95_latency_ms:.1f}ms, "
                      f"QPS={result.throughput_qps:.1f}")
    else:
        logger.error("No benchmark results generated")

if __name__ == '__main__':
    main() 