#!/usr/bin/env python3
"""
AWS Neuron Compiler
Compiles ONNX/PyTorch models to optimized Neuron models for AWS Inferentia/Trainium
"""

import argparse
import json
import logging
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

import torch
import torch_neuronx
import numpy as np
import onnx
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NeuronCompiler:
    """Handles Neuron compilation for AWS Inferentia/Trainium"""
    
    def __init__(self, output_dir: str = "models/neuron", s3_bucket: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3') if s3_bucket else None
        
        # Check Neuron environment
        self._check_neuron_environment()
    
    def _check_neuron_environment(self):
        """Check if Neuron SDK is properly installed"""
        try:
            # Check if neuron-cc is available
            result = subprocess.run(['neuron-cc', '--version'], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"Neuron compiler version: {result.stdout.strip()}")
            
            # Check torch-neuronx
            logger.info(f"torch-neuronx version: {torch_neuronx.__version__}")
            
        except subprocess.CalledProcessError:
            logger.warning("Neuron compiler not found. Some features may not work.")
        except Exception as e:
            logger.warning(f"Neuron environment check failed: {str(e)}")
    
    def compile_pytorch_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...],
        compiler_workdir: Optional[str] = None,
        compiler_args: Optional[List[str]] = None,
        optimization_level: str = "2",
        auto_cast: bool = True,
        dynamic_batch_size: bool = False
    ) -> Dict[str, Any]:
        """Compile PyTorch model for Neuron"""
        logger.info(f"Compiling PyTorch model for Neuron: {model_name}")
        
        start_time = time.time()
        
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Create example input
            example_inputs = torch.randn(*input_shape)
            
            # Prepare compiler arguments
            if compiler_args is None:
                compiler_args = [
                    f"--opt-level={optimization_level}",
                    "--enable-saturate-infinity",
                    "--enable-mixed-precision-accumulation"
                ]
            
            if auto_cast:
                compiler_args.append("--auto-cast=fp16")
            
            if dynamic_batch_size:
                compiler_args.append("--dynamic-batch-size")
            
            # Set compiler workdir
            if compiler_workdir is None:
                compiler_workdir = str(self.output_dir / f"{model_name}_workdir")
            
            os.environ['NEURON_CC_FLAGS'] = ' '.join(compiler_args)
            os.environ['NEURON_COMPILE_CACHE_URL'] = compiler_workdir
            
            # Compile model
            logger.info("Starting Neuron compilation (this may take several minutes)...")
            
            compiled_model = torch_neuronx.trace(
                model,
                example_inputs,
                compiler_workdir=compiler_workdir,
                compiler_args=compiler_args
            )
            
            # Save compiled model
            model_path = self.output_dir / f"{model_name}.pt"
            torch.jit.save(compiled_model, str(model_path))
            
            compile_time = time.time() - start_time
            
            # Get compilation info
            compilation_info = self._get_compilation_info(compiler_workdir)
            
            # Validate compiled model
            validation_result = self._validate_neuron_model(
                compiled_model, example_inputs, model, model_name
            )
            
            # Get model size and details
            model_stats = self._get_model_stats(model_path, compiled_model)
            
            # Upload to S3 if configured
            s3_path = None
            if self.s3_bucket:
                s3_path = self._upload_to_s3(model_path, model_name)
            
            result = {
                'model_name': model_name,
                'model_path': str(model_path),
                's3_path': s3_path,
                'input_shape': input_shape,
                'compile_time_seconds': compile_time,
                'model_size_mb': model_path.stat().st_size / (1024 * 1024),
                'compilation_info': compilation_info,
                'validation': validation_result,
                'model_stats': model_stats,
                'compiler_args': compiler_args,
                'optimization_level': optimization_level,
                'compile_time': time.time()
            }
            
            logger.info(f"Successfully compiled {model_name} for Neuron ({compile_time:.1f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to compile {model_name} for Neuron: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def compile_onnx_model(
        self,
        onnx_path: str,
        model_name: str,
        input_shape: Tuple[int, ...],
        compiler_workdir: Optional[str] = None,
        compiler_args: Optional[List[str]] = None,
        optimization_level: str = "2"
    ) -> Dict[str, Any]:
        """Compile ONNX model for Neuron"""
        logger.info(f"Compiling ONNX model for Neuron: {model_name}")
        
        start_time = time.time()
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert ONNX to PyTorch
            # Note: This is a simplified conversion. Real implementation might use onnx2torch
            logger.info("Converting ONNX to PyTorch for Neuron compilation...")
            
            # For now, we'll simulate this process
            # In practice, you'd use a proper ONNX to PyTorch converter
            pytorch_model = self._convert_onnx_to_pytorch(onnx_model, model_name)
            
            # Compile the converted model
            result = self.compile_pytorch_model(
                model=pytorch_model,
                model_name=f"{model_name}_from_onnx",
                input_shape=input_shape,
                compiler_workdir=compiler_workdir,
                compiler_args=compiler_args,
                optimization_level=optimization_level
            )
            
            result['source_format'] = 'onnx'
            result['onnx_path'] = onnx_path
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compile ONNX model {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def compile_huggingface_model(
        self,
        model_name_or_path: str,
        model_name: str,
        sequence_length: int = 512,
        batch_size: int = 1,
        num_neuron_cores: int = 1,
        optimization_level: str = "2",
        auto_cast: bool = True
    ) -> Dict[str, Any]:
        """Compile HuggingFace transformer model for Neuron"""
        logger.info(f"Compiling HuggingFace model for Neuron: {model_name}")
        
        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig
            
            # Load model and tokenizer
            config = AutoConfig.from_pretrained(model_name_or_path)
            model = AutoModel.from_pretrained(model_name_or_path, torchscript=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Prepare for compilation
            model.eval()
            
            # Create example inputs for tracing
            example_inputs = {
                'input_ids': torch.ones((batch_size, sequence_length), dtype=torch.long),
                'attention_mask': torch.ones((batch_size, sequence_length), dtype=torch.long)
            }
            
            # Prepare compiler arguments for transformers
            compiler_args = [
                f"--opt-level={optimization_level}",
                f"--num-neuroncores={num_neuron_cores}",
                "--enable-saturate-infinity",
                "--enable-mixed-precision-accumulation"
            ]
            
            if auto_cast:
                compiler_args.append("--auto-cast=fp16")
            
            # Set environment
            compiler_workdir = str(self.output_dir / f"{model_name}_hf_workdir")
            os.environ['NEURON_CC_FLAGS'] = ' '.join(compiler_args)
            
            # Compile model
            logger.info("Compiling HuggingFace transformer (this may take 10-20 minutes)...")
            
            # Use torch_neuronx for transformer compilation
            compiled_model = torch_neuronx.trace(
                model,
                example_inputs,
                compiler_workdir=compiler_workdir,
                compiler_args=compiler_args
            )
            
            # Save compiled model
            model_path = self.output_dir / f"{model_name}_hf.pt"
            torch.jit.save(compiled_model, str(model_path))
            
            # Validate compiled model
            validation_result = self._validate_transformer_model(
                compiled_model, example_inputs, model, tokenizer
            )
            
            # Get model stats
            model_stats = self._get_model_stats(model_path, compiled_model)
            model_stats['sequence_length'] = sequence_length
            model_stats['num_neuron_cores'] = num_neuron_cores
            
            # Upload to S3 if configured
            s3_path = None
            if self.s3_bucket:
                s3_path = self._upload_to_s3(model_path, f"{model_name}_hf")
            
            result = {
                'model_name': f"{model_name}_hf",
                'model_path': str(model_path),
                's3_path': s3_path,
                'sequence_length': sequence_length,
                'batch_size': batch_size,
                'num_neuron_cores': num_neuron_cores,
                'model_size_mb': model_path.stat().st_size / (1024 * 1024),
                'validation': validation_result,
                'model_stats': model_stats,
                'compiler_args': compiler_args,
                'framework': 'huggingface',
                'compile_time': time.time()
            }
            
            logger.info(f"Successfully compiled HuggingFace model {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to compile HuggingFace model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def compile_multiple_configurations(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shapes: List[Tuple[int, ...]],
        optimization_levels: List[str] = ["1", "2", "3"],
        auto_cast_options: List[bool] = [True, False]
    ) -> List[Dict[str, Any]]:
        """Compile model with multiple configurations"""
        results = []
        
        for opt_level in optimization_levels:
            for auto_cast in auto_cast_options:
                for i, input_shape in enumerate(input_shapes):
                    config_name = f"{model_name}_opt{opt_level}_cast{auto_cast}_shape{i}"
                    
                    try:
                        result = self.compile_pytorch_model(
                            model=model,
                            model_name=config_name,
                            input_shape=input_shape,
                            optimization_level=opt_level,
                            auto_cast=auto_cast
                        )
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Failed to compile {config_name}: {str(e)}")
                        results.append({
                            'model_name': config_name,
                            'optimization_level': opt_level,
                            'auto_cast': auto_cast,
                            'input_shape': input_shape,
                            'error': str(e),
                            'success': False
                        })
        
        return results
    
    def _convert_onnx_to_pytorch(self, onnx_model, model_name: str) -> torch.nn.Module:
        """Convert ONNX model to PyTorch (simplified implementation)"""
        # This is a placeholder. In practice, you'd use a proper converter like onnx2torch
        logger.info(f"Converting ONNX model {model_name} to PyTorch...")
        
        # For demonstration, create a simple model based on the ONNX graph
        # Real implementation would parse the ONNX graph and create equivalent PyTorch layers
        
        class SimpleONNXModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # This is a placeholder - would be built from ONNX graph
                self.layer1 = torch.nn.Linear(512, 256)
                self.layer2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(256, 128)
                
            def forward(self, x):
                x = self.layer1(x.view(x.size(0), -1))
                x = self.layer2(x)
                x = self.layer3(x)
                return x
        
        return SimpleONNXModel()
    
    def _get_compilation_info(self, compiler_workdir: str) -> Dict[str, Any]:
        """Extract compilation information from Neuron compiler output"""
        try:
            workdir_path = Path(compiler_workdir)
            
            info = {
                'workdir': compiler_workdir,
                'compiler_artifacts': [],
                'logs': None
            }
            
            if workdir_path.exists():
                # List compilation artifacts
                for file_path in workdir_path.rglob("*"):
                    if file_path.is_file():
                        info['compiler_artifacts'].append({
                            'name': file_path.name,
                            'path': str(file_path),
                            'size_bytes': file_path.stat().st_size
                        })
                
                # Try to read compilation logs
                log_files = list(workdir_path.glob("*.log"))
                if log_files:
                    try:
                        with open(log_files[0], 'r') as f:
                            info['logs'] = f.read()
                    except Exception:
                        pass
            
            return info
            
        except Exception as e:
            logger.warning(f"Failed to get compilation info: {str(e)}")
            return {'error': str(e)}
    
    def _validate_neuron_model(
        self,
        compiled_model,
        example_inputs: torch.Tensor,
        original_model: torch.nn.Module,
        model_name: str
    ) -> Dict[str, Any]:
        """Validate compiled Neuron model"""
        try:
            logger.info(f"Validating Neuron model: {model_name}")
            
            # Run inference with compiled model
            compiled_model.eval()
            with torch.no_grad():
                neuron_output = compiled_model(example_inputs)
            
            # Run inference with original model
            original_model.eval()
            with torch.no_grad():
                original_output = original_model(example_inputs)
            
            # Compare outputs
            if isinstance(neuron_output, tuple):
                neuron_output = neuron_output[0]
            if isinstance(original_output, tuple):
                original_output = original_output[0]
            
            # Calculate difference
            max_diff = torch.abs(neuron_output - original_output).max().item()
            mean_diff = torch.abs(neuron_output - original_output).mean().item()
            
            # Check if validation passes
            validation_passed = max_diff < 1e-3  # Tolerance for Neuron
            
            return {
                'passed': validation_passed,
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'output_shape': list(neuron_output.shape),
                'inference_successful': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Validation failed for {model_name}: {str(e)}")
            return {
                'passed': False,
                'inference_successful': False,
                'error': str(e)
            }
    
    def _validate_transformer_model(
        self,
        compiled_model,
        example_inputs: Dict[str, torch.Tensor],
        original_model: torch.nn.Module,
        tokenizer
    ) -> Dict[str, Any]:
        """Validate compiled transformer model"""
        try:
            logger.info("Validating compiled transformer model...")
            
            # Run inference with compiled model
            with torch.no_grad():
                neuron_output = compiled_model(**example_inputs)
            
            # Run inference with original model
            with torch.no_grad():
                original_output = original_model(**example_inputs)
            
            # Extract last hidden state
            if hasattr(neuron_output, 'last_hidden_state'):
                neuron_tensor = neuron_output.last_hidden_state
            else:
                neuron_tensor = neuron_output[0] if isinstance(neuron_output, tuple) else neuron_output
            
            if hasattr(original_output, 'last_hidden_state'):
                original_tensor = original_output.last_hidden_state
            else:
                original_tensor = original_output[0] if isinstance(original_output, tuple) else original_output
            
            # Compare outputs
            max_diff = torch.abs(neuron_tensor - original_tensor).max().item()
            mean_diff = torch.abs(neuron_tensor - original_tensor).mean().item()
            
            validation_passed = max_diff < 1e-2  # More relaxed tolerance for transformers
            
            return {
                'passed': validation_passed,
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'output_shape': list(neuron_tensor.shape),
                'inference_successful': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Transformer validation failed: {str(e)}")
            return {
                'passed': False,
                'inference_successful': False,
                'error': str(e)
            }
    
    def _get_model_stats(self, model_path: Path, compiled_model) -> Dict[str, Any]:
        """Get detailed model statistics"""
        try:
            stats = {
                'file_size_mb': model_path.stat().st_size / (1024 * 1024),
                'file_path': str(model_path)
            }
            
            # Try to get parameter count
            try:
                total_params = sum(p.numel() for p in compiled_model.parameters())
                stats['parameter_count'] = total_params
            except Exception:
                stats['parameter_count'] = None
            
            # Get model info if available
            try:
                model_info = str(compiled_model).split('\n')[:10]  # First 10 lines
                stats['model_summary'] = model_info
            except Exception:
                stats['model_summary'] = None
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get model stats: {str(e)}")
            return {}
    
    def _upload_to_s3(self, local_path: Path, model_name: str) -> Optional[str]:
        """Upload compiled model to S3"""
        try:
            if not self.s3_client:
                return None
            
            s3_key = f"models/neuron/{model_name}/{local_path.name}"
            
            self.s3_client.upload_file(
                str(local_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'Metadata': {
                        'model_name': model_name,
                        'compile_time': str(int(time.time())),
                        'framework': 'neuron'
                    }
                }
            )
            
            s3_path = f"s3://{self.s3_bucket}/{s3_key}"
            logger.info(f"Uploaded model to {s3_path}")
            return s3_path
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            return None
    
    def benchmark_neuron_model(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """Benchmark compiled Neuron model"""
        logger.info(f"Benchmarking Neuron model: {model_path}")
        
        try:
            # Load compiled model
            compiled_model = torch.jit.load(model_path)
            compiled_model.eval()
            
            # Create example input
            example_input = torch.randn(*input_shape)
            
            # Warmup
            logger.info(f"Warming up with {warmup_iterations} iterations...")
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    _ = compiled_model(example_input)
            
            # Benchmark
            logger.info(f"Running benchmark with {num_iterations} iterations...")
            latencies = []
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    start_time = time.time()
                    _ = compiled_model(example_input)
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
                'batch_size': input_shape[0],
                'input_shape': input_shape
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            return {'error': str(e)}

def compile_models_from_directory(
    model_dir: str,
    output_dir: str = "models/neuron",
    s3_bucket: Optional[str] = None,
    optimization_levels: List[str] = ["2"]
) -> List[Dict[str, Any]]:
    """Compile all models in a directory for Neuron"""
    
    compiler = NeuronCompiler(output_dir, s3_bucket)
    model_dir = Path(model_dir)
    results = []
    
    # Process ONNX models
    for onnx_file in model_dir.glob("*.onnx"):
        model_name = onnx_file.stem
        logger.info(f"Compiling ONNX model for Neuron: {model_name}")
        
        try:
            # Determine input shape from ONNX model
            onnx_model = onnx.load(str(onnx_file))
            input_shape = []
            for input_tensor in onnx_model.graph.input:
                shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
                input_shape = tuple(shape)
                break
            
            if not input_shape:
                input_shape = (1, 3, 224, 224)  # Default shape
            
            for opt_level in optimization_levels:
                result = compiler.compile_onnx_model(
                    onnx_path=str(onnx_file),
                    model_name=f"{model_name}_opt{opt_level}",
                    input_shape=input_shape,
                    optimization_level=opt_level
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Failed to compile {model_name}: {str(e)}")
            results.append({
                'model_name': model_name,
                'error': str(e),
                'success': False
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compile models for AWS Neuron')
    parser.add_argument('--model-path', help='Path to model file (PyTorch .pt or ONNX .onnx)')
    parser.add_argument('--model-dir', help='Directory containing models to compile')
    parser.add_argument('--output-dir', default='models/neuron', help='Output directory')
    parser.add_argument('--model-name', help='Name for the compiled model')
    parser.add_argument('--input-shape', help='Input shape (comma-separated)', default='1,3,224,224')
    parser.add_argument('--optimization-level', choices=['0', '1', '2', '3'], default='2', help='Optimization level')
    parser.add_argument('--auto-cast', action='store_true', default=True, help='Enable auto cast to FP16')
    parser.add_argument('--s3-bucket', help='S3 bucket for model upload')
    parser.add_argument('--compile-all', action='store_true', help='Compile all models in directory')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark compiled models')
    parser.add_argument('--huggingface-model', help='HuggingFace model name or path')
    parser.add_argument('--sequence-length', type=int, default=512, help='Sequence length for transformers')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-neuron-cores', type=int, default=1, help='Number of Neuron cores to use')
    
    args = parser.parse_args()
    
    if args.compile_all and args.model_dir:
        logger.info("Compiling all models for Neuron...")
        results = compile_models_from_directory(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            s3_bucket=args.s3_bucket,
            optimization_levels=[args.optimization_level]
        )
        
        # Save results
        results_file = Path(args.output_dir) / 'neuron_compile_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Compilation results saved to {results_file}")
        
        # Print summary
        successful = sum(1 for r in results if r.get('validation', {}).get('passed', False))
        print(f"\n=== Neuron Compilation Summary ===")
        print(f"Total models: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        for result in results:
            if result.get('success', True):
                status = "[PASS]" if result.get('validation', {}).get('passed', False) else "[FAIL]"
                size = result.get('model_size_mb', 0)
                compile_time = result.get('compile_time_seconds', 0)
                print(f"{status} {result['model_name']}: {size:.1f} MB ({compile_time:.1f}s)")
    
    elif args.huggingface_model and args.model_name:
        logger.info(f"Compiling HuggingFace model: {args.huggingface_model}")
        compiler = NeuronCompiler(args.output_dir, args.s3_bucket)
        
        result = compiler.compile_huggingface_model(
            model_name_or_path=args.huggingface_model,
            model_name=args.model_name,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            num_neuron_cores=args.num_neuron_cores,
            optimization_level=args.optimization_level,
            auto_cast=args.auto_cast
        )
        
        print(f"Compilation completed: {result}")
        
        if args.benchmark:
            input_shape = (args.batch_size, args.sequence_length)
            benchmark_result = compiler.benchmark_neuron_model(
                result['model_path'], input_shape
            )
            print(f"Benchmark results: {benchmark_result}")
    
    elif args.model_path and args.model_name:
        logger.info(f"Compiling single model: {args.model_name}")
        compiler = NeuronCompiler(args.output_dir, args.s3_bucket)
        
        input_shape = tuple(map(int, args.input_shape.split(',')))
        
        if args.model_path.endswith('.onnx'):
            result = compiler.compile_onnx_model(
                onnx_path=args.model_path,
                model_name=args.model_name,
                input_shape=input_shape,
                optimization_level=args.optimization_level
            )
        elif args.model_path.endswith('.pt') or args.model_path.endswith('.pth'):
            model = torch.jit.load(args.model_path)
            result = compiler.compile_pytorch_model(
                model=model,
                model_name=args.model_name,
                input_shape=input_shape,
                optimization_level=args.optimization_level,
                auto_cast=args.auto_cast
            )
        else:
            raise ValueError("Unsupported model format. Use .pt, .pth, or .onnx")
        
        print(f"Compilation completed: {result}")
        
        if args.benchmark:
            benchmark_result = compiler.benchmark_neuron_model(
                result['model_path'], input_shape
            )
            print(f"Benchmark results: {benchmark_result}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 