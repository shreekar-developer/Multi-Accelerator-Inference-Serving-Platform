#!/usr/bin/env python3
"""
TensorRT Engine Builder
Compiles ONNX models to optimized TensorRT engines for GPU inference
"""

import argparse
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TensorRTBuilder:
    """Handles TensorRT engine compilation from ONNX models"""
    
    def __init__(self, output_dir: str = "models/tensorrt", s3_bucket: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3') if s3_bucket else None
        
        # Initialize TensorRT logger and builder
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.builder = trt.Builder(self.trt_logger)
        
    def build_engine(
        self,
        onnx_path: str,
        engine_name: str,
        precision: str = "fp16",
        max_batch_size: int = 32,
        max_workspace_size: int = 1 << 30,  # 1GB
        optimization_level: int = 3,
        dynamic_shapes: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = None,
        calibration_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Build TensorRT engine from ONNX model"""
        logger.info(f"Building TensorRT engine: {engine_name} with precision: {precision}")
        
        start_time = time.time()
        
        try:
            # Create network and parser
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = self.builder.create_network(network_flags)
            parser = trt.OnnxParser(network, self.trt_logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    error_msgs = []
                    for i in range(parser.num_errors):
                        error_msgs.append(str(parser.get_error(i)))
                    raise RuntimeError(f"Failed to parse ONNX model: {'; '.join(error_msgs)}")
            
            # Configure builder
            config = self.builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            # Set precision mode
            if precision == "fp16":
                if not self.builder.platform_has_fast_fp16:
                    logger.warning("FP16 not supported on this platform, falling back to FP32")
                    precision = "fp32"
                else:
                    config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8":
                if not self.builder.platform_has_fast_int8:
                    logger.warning("INT8 not supported on this platform, falling back to FP16")
                    precision = "fp16"
                    config.set_flag(trt.BuilderFlag.FP16)
                else:
                    config.set_flag(trt.BuilderFlag.INT8)
                    if calibration_data is not None:
                        config.int8_calibrator = self._create_int8_calibrator(calibration_data)
            
            # Set optimization level
            if hasattr(config, 'builder_optimization_level'):
                config.builder_optimization_level = optimization_level
            
            # Configure dynamic shapes if provided
            if dynamic_shapes:
                profile = self.builder.create_optimization_profile()
                for input_name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
                    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)
            else:
                # Set max batch size for implicit batch mode
                if max_batch_size > 1:
                    self.builder.max_batch_size = max_batch_size
            
            # Build engine
            logger.info("Building TensorRT engine (this may take several minutes)...")
            engine = self.builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Serialize engine
            engine_path = self.output_dir / f"{engine_name}.trt"
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            build_time = time.time() - start_time
            
            # Get engine info
            engine_info = self._get_engine_info(engine)
            
            # Validate engine
            validation_result = self._validate_engine(engine_path, onnx_path)
            
            # Upload to S3 if configured
            s3_path = None
            if self.s3_bucket:
                s3_path = self._upload_to_s3(engine_path, engine_name)
            
            result = {
                'engine_name': engine_name,
                'engine_path': str(engine_path),
                's3_path': s3_path,
                'precision': precision,
                'max_batch_size': max_batch_size,
                'build_time_seconds': build_time,
                'engine_size_mb': engine_path.stat().st_size / (1024 * 1024),
                'validation': validation_result,
                'engine_info': engine_info,
                'build_time': time.time()
            }
            
            logger.info(f"Successfully built TensorRT engine: {engine_path} ({build_time:.1f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to build TensorRT engine: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def build_multiple_precisions(
        self,
        onnx_path: str,
        engine_name: str,
        precisions: List[str] = ["fp32", "fp16", "int8"],
        max_batch_sizes: List[int] = [1, 8, 32],
        calibration_data: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Build engines with multiple precision modes and batch sizes"""
        results = []
        
        for precision in precisions:
            for batch_size in max_batch_sizes:
                engine_variant_name = f"{engine_name}_{precision}_bs{batch_size}"
                
                try:
                    result = self.build_engine(
                        onnx_path=onnx_path,
                        engine_name=engine_variant_name,
                        precision=precision,
                        max_batch_size=batch_size,
                        calibration_data=calibration_data if precision == "int8" else None
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to build {engine_variant_name}: {str(e)}")
                    results.append({
                        'engine_name': engine_variant_name,
                        'precision': precision,
                        'max_batch_size': batch_size,
                        'error': str(e),
                        'success': False
                    })
        
        return results
    
    def build_dynamic_shapes_engine(
        self,
        onnx_path: str,
        engine_name: str,
        input_specs: Dict[str, Dict[str, Tuple[int, ...]]],
        precision: str = "fp16"
    ) -> Dict[str, Any]:
        """Build engine with dynamic input shapes (for NLP models)"""
        logger.info(f"Building dynamic shapes TensorRT engine: {engine_name}")
        
        # Convert input specs to dynamic shapes format
        dynamic_shapes = {}
        for input_name, shapes in input_specs.items():
            dynamic_shapes[input_name] = (
                shapes['min'],
                shapes['opt'],
                shapes['max']
            )
        
        return self.build_engine(
            onnx_path=onnx_path,
            engine_name=f"{engine_name}_dynamic",
            precision=precision,
            dynamic_shapes=dynamic_shapes
        )
    
    def _create_int8_calibrator(self, calibration_data: np.ndarray):
        """Create INT8 calibrator for quantization"""
        
        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, data):
                trt.IInt8EntropyCalibrator2.__init__(self)
                self.data = data
                self.current_index = 0
                self.batch_size = data.shape[0] if len(data.shape) > 0 else 1
                
                # Allocate device memory
                self.device_input = cuda.mem_alloc(data.nbytes)
                
            def get_batch_size(self):
                return self.batch_size
                
            def get_batch(self, names):
                if self.current_index < len(self.data):
                    batch = self.data[self.current_index:self.current_index + self.batch_size]
                    cuda.memcpy_htod(self.device_input, batch.astype(np.float32))
                    self.current_index += self.batch_size
                    return [int(self.device_input)]
                else:
                    return None
                    
            def read_calibration_cache(self):
                # Return None to indicate no cache
                return None
                
            def write_calibration_cache(self, cache):
                # Save calibration cache
                pass
        
        return Calibrator(calibration_data)
    
    def _get_engine_info(self, engine) -> Dict[str, Any]:
        """Get detailed engine information"""
        try:
            info = {
                'num_bindings': engine.num_bindings,
                'max_batch_size': engine.max_batch_size,
                'has_implicit_batch_dimension': engine.has_implicit_batch_dimension,
                'bindings': []
            }
            
            for i in range(engine.num_bindings):
                binding_name = engine.get_binding_name(i)
                binding_shape = engine.get_binding_shape(i)
                binding_dtype = trt.nptype(engine.get_binding_dtype(i))
                is_input = engine.binding_is_input(i)
                
                info['bindings'].append({
                    'name': binding_name,
                    'shape': list(binding_shape),
                    'dtype': str(binding_dtype),
                    'is_input': is_input
                })
            
            return info
            
        except Exception as e:
            logger.warning(f"Failed to get engine info: {str(e)}")
            return {}
    
    def _validate_engine(self, engine_path: Path, onnx_path: str) -> Dict[str, Any]:
        """Validate TensorRT engine"""
        try:
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            if engine is None:
                return {
                    'passed': False,
                    'error': 'Failed to deserialize engine'
                }
            
            # Create execution context
            context = engine.create_execution_context()
            
            if context is None:
                return {
                    'passed': False,
                    'error': 'Failed to create execution context'
                }
            
            # Basic validation - ensure engine can be loaded and context created
            validation_result = {
                'passed': True,
                'engine_loaded': True,
                'context_created': True,
                'num_bindings': engine.num_bindings,
                'error': None
            }
            
            # Try to run a simple inference test if possible
            try:
                validation_result.update(self._run_inference_test(engine, context))
            except Exception as e:
                logger.warning(f"Inference test failed: {str(e)}")
                validation_result['inference_test'] = False
            
            return validation_result
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _run_inference_test(self, engine, context) -> Dict[str, Any]:
        """Run a basic inference test on the engine"""
        try:
            # Allocate device memory for inputs and outputs
            bindings = []
            for i in range(engine.num_bindings):
                binding_shape = engine.get_binding_shape(i)
                binding_dtype = trt.nptype(engine.get_binding_dtype(i))
                
                # Calculate size
                size = trt.volume(binding_shape) * engine.max_batch_size
                
                # Allocate device memory
                device_mem = cuda.mem_alloc(size * binding_dtype().itemsize)
                bindings.append(int(device_mem))
                
                # Initialize input with random data
                if engine.binding_is_input(i):
                    host_input = np.random.random((engine.max_batch_size,) + tuple(binding_shape[1:])).astype(binding_dtype)
                    cuda.memcpy_htod(device_mem, host_input)
            
            # Run inference
            context.execute_v2(bindings)
            
            return {
                'inference_test': True,
                'inference_successful': True
            }
            
        except Exception as e:
            return {
                'inference_test': False,
                'inference_error': str(e)
            }
    
    def _upload_to_s3(self, local_path: Path, engine_name: str) -> Optional[str]:
        """Upload engine to S3"""
        try:
            if not self.s3_client:
                return None
            
            s3_key = f"models/tensorrt/{engine_name}/{local_path.name}"
            
            self.s3_client.upload_file(
                str(local_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'Metadata': {
                        'engine_name': engine_name,
                        'build_time': str(int(time.time())),
                        'framework': 'tensorrt'
                    }
                }
            )
            
            s3_path = f"s3://{self.s3_bucket}/{s3_key}"
            logger.info(f"Uploaded engine to {s3_path}")
            return s3_path
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            return None
    
    def benchmark_engine(self, engine_path: str, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark TensorRT engine performance"""
        logger.info(f"Benchmarking engine: {engine_path}")
        
        try:
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            # Prepare bindings
            bindings = []
            input_shapes = []
            
            for i in range(engine.num_bindings):
                binding_shape = engine.get_binding_shape(i)
                binding_dtype = trt.nptype(engine.get_binding_dtype(i))
                
                size = trt.volume(binding_shape) * engine.max_batch_size
                device_mem = cuda.mem_alloc(size * binding_dtype().itemsize)
                bindings.append(int(device_mem))
                
                if engine.binding_is_input(i):
                    input_shapes.append((engine.max_batch_size,) + tuple(binding_shape[1:]))
                    host_input = np.random.random(input_shapes[-1]).astype(binding_dtype)
                    cuda.memcpy_htod(device_mem, host_input)
            
            # Warmup
            for _ in range(10):
                context.execute_v2(bindings)
            cuda.Context.synchronize()
            
            # Benchmark
            latencies = []
            for _ in range(num_iterations):
                start_time = time.time()
                context.execute_v2(bindings)
                cuda.Context.synchronize()
                latencies.append((time.time() - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            latencies = np.array(latencies)
            
            return {
                'mean_latency_ms': float(np.mean(latencies)),
                'p50_latency_ms': float(np.percentile(latencies, 50)),
                'p95_latency_ms': float(np.percentile(latencies, 95)),
                'p99_latency_ms': float(np.percentile(latencies, 99)),
                'min_latency_ms': float(np.min(latencies)),
                'max_latency_ms': float(np.max(latencies)),
                'std_latency_ms': float(np.std(latencies)),
                'throughput_fps': 1000.0 / np.mean(latencies),
                'num_iterations': num_iterations,
                'batch_size': engine.max_batch_size
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            return {'error': str(e)}

def create_calibration_data(model_type: str, input_shape: Tuple[int, ...], num_samples: int = 100) -> np.ndarray:
    """Create calibration data for INT8 quantization"""
    logger.info(f"Creating calibration data for {model_type}")
    
    if model_type == "vision":
        # Create realistic image data (normalized)
        data = np.random.randn(num_samples, *input_shape[1:]).astype(np.float32)
        data = np.clip(data * 0.229 + 0.485, 0, 1)  # ImageNet normalization
    elif model_type == "nlp":
        # Create realistic token IDs
        vocab_size = 30522  # BERT vocab size
        data = np.random.randint(0, vocab_size, (num_samples, *input_shape[1:]), dtype=np.int32)
    else:
        # Generic random data
        data = np.random.randn(num_samples, *input_shape[1:]).astype(np.float32)
    
    return data

def build_engines_from_onnx_directory(
    onnx_dir: str,
    output_dir: str = "models/tensorrt",
    s3_bucket: Optional[str] = None,
    precisions: List[str] = ["fp16", "fp32"],
    batch_sizes: List[int] = [1, 8, 32]
) -> List[Dict[str, Any]]:
    """Build TensorRT engines for all ONNX models in a directory"""
    
    builder = TensorRTBuilder(output_dir, s3_bucket)
    onnx_dir = Path(onnx_dir)
    results = []
    
    for onnx_file in onnx_dir.glob("*.onnx"):
        if onnx_file.name.endswith("_optimized.onnx"):
            continue  # Skip optimized versions
        
        model_name = onnx_file.stem
        logger.info(f"Building TensorRT engines for {model_name}")
        
        try:
            # Determine model type for calibration data
            model_type = "vision"
            if "bert" in model_name.lower() or "distil" in model_name.lower():
                model_type = "nlp"
            
            # Create calibration data for INT8 if needed
            calibration_data = None
            if "int8" in precisions:
                # Load ONNX model to get input shape
                onnx_model = onnx.load(str(onnx_file))
                input_shape = []
                for input_tensor in onnx_model.graph.input:
                    shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
                    input_shape = tuple(shape)
                    break
                
                if input_shape:
                    calibration_data = create_calibration_data(model_type, input_shape)
            
            # Build engines with multiple configurations
            engine_results = builder.build_multiple_precisions(
                onnx_path=str(onnx_file),
                engine_name=model_name,
                precisions=precisions,
                max_batch_sizes=batch_sizes,
                calibration_data=calibration_data
            )
            
            results.extend(engine_results)
            
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {str(e)}")
            results.append({
                'model_name': model_name,
                'error': str(e),
                'success': False
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Build TensorRT engines from ONNX models')
    parser.add_argument('--onnx-path', help='Path to ONNX model file')
    parser.add_argument('--onnx-dir', help='Directory containing ONNX models')
    parser.add_argument('--output-dir', default='models/tensorrt', help='Output directory for TensorRT engines')
    parser.add_argument('--engine-name', help='Name for the TensorRT engine')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16', help='Precision mode')
    parser.add_argument('--precisions', nargs='+', choices=['fp32', 'fp16', 'int8'], default=['fp16'], help='Multiple precision modes')
    parser.add_argument('--max-batch-size', type=int, default=32, help='Maximum batch size')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 32], help='Multiple batch sizes')
    parser.add_argument('--max-workspace-size', type=int, default=1<<30, help='Maximum workspace size in bytes')
    parser.add_argument('--s3-bucket', help='S3 bucket for engine upload')
    parser.add_argument('--build-all', action='store_true', help='Build engines for all ONNX models in directory')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark built engines')
    parser.add_argument('--dynamic-shapes', help='JSON file with dynamic shape specifications')
    
    args = parser.parse_args()
    
    if args.build_all and args.onnx_dir:
        logger.info("Building TensorRT engines for all ONNX models...")
        results = build_engines_from_onnx_directory(
            onnx_dir=args.onnx_dir,
            output_dir=args.output_dir,
            s3_bucket=args.s3_bucket,
            precisions=args.precisions,
            batch_sizes=args.batch_sizes
        )
        
        # Save results
        results_file = Path(args.output_dir) / 'build_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Build results saved to {results_file}")
        
        # Print summary
        successful = sum(1 for r in results if r.get('validation', {}).get('passed', False))
        print(f"\n=== Build Summary ===")
        print(f"Total engines: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        for result in results:
            if result.get('success', True):
                status = "[PASS]" if result.get('validation', {}).get('passed', False) else "[FAIL]"
                size = result.get('engine_size_mb', 0)
                build_time = result.get('build_time_seconds', 0)
                print(f"{status} {result['engine_name']}: {size:.1f} MB ({build_time:.1f}s)")
    
    elif args.onnx_path and args.engine_name:
        logger.info(f"Building single TensorRT engine: {args.engine_name}")
        builder = TensorRTBuilder(args.output_dir, args.s3_bucket)
        
        # Handle dynamic shapes if provided
        dynamic_shapes = None
        if args.dynamic_shapes:
            with open(args.dynamic_shapes, 'r') as f:
                dynamic_shapes_config = json.load(f)
                dynamic_shapes = {}
                for input_name, shapes in dynamic_shapes_config.items():
                    dynamic_shapes[input_name] = (
                        tuple(shapes['min']),
                        tuple(shapes['opt']),
                        tuple(shapes['max'])
                    )
        
        result = builder.build_engine(
            onnx_path=args.onnx_path,
            engine_name=args.engine_name,
            precision=args.precision,
            max_batch_size=args.max_batch_size,
            max_workspace_size=args.max_workspace_size,
            dynamic_shapes=dynamic_shapes
        )
        
        print(f"Build completed: {result}")
        
        # Benchmark if requested
        if args.benchmark:
            benchmark_result = builder.benchmark_engine(result['engine_path'])
            print(f"Benchmark results: {benchmark_result}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 