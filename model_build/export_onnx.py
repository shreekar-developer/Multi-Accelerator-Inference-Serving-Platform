#!/usr/bin/env python3
"""
ONNX Export Pipeline
Converts PyTorch/TensorFlow models to ONNX format with optimization and validation
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision.models as models
import tensorflow as tf
import onnx
import onnxruntime as ort
import numpy as np
from transformers import (
    AutoModel, AutoTokenizer, DistilBertModel, DistilBertTokenizer,
    BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
)
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ONNXExporter:
    """Handles model export to ONNX format with validation and optimization"""
    
    def __init__(self, output_dir: str = "models/onnx", s3_bucket: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3') if s3_bucket else None
        
    def export_pytorch_model(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...],
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """Export PyTorch model to ONNX"""
        logger.info(f"Exporting PyTorch model: {model_name}")
        
        try:
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            
            # Set model to eval mode
            model.eval()
            
            # Export path
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes or {
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
            
            # Validate exported model
            validation_result = self._validate_onnx_model(onnx_path, dummy_input, model)
            
            # Optimize if requested
            if optimize:
                optimized_path = self._optimize_onnx_model(onnx_path)
                onnx_path = optimized_path
            
            # Get model info
            model_info = self._get_model_info(onnx_path)
            
            # Upload to S3 if configured
            s3_path = None
            if self.s3_bucket:
                s3_path = self._upload_to_s3(onnx_path, model_name)
            
            result = {
                'model_name': model_name,
                'framework': 'pytorch',
                'onnx_path': str(onnx_path),
                's3_path': s3_path,
                'input_shape': input_shape,
                'model_size_mb': onnx_path.stat().st_size / (1024 * 1024),
                'validation': validation_result,
                'model_info': model_info,
                'export_time': time.time()
            }
            
            logger.info(f"Successfully exported {model_name} to {onnx_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to export {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def export_tensorflow_model(
        self,
        model_path: str,
        model_name: str,
        input_shape: Tuple[int, ...],
        optimize: bool = True
    ) -> Dict[str, Any]:
        """Export TensorFlow model to ONNX"""
        logger.info(f"Exporting TensorFlow model: {model_name}")
        
        try:
            import tf2onnx
            
            # Load TensorFlow model
            model = tf.saved_model.load(model_path)
            
            # Convert to ONNX
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
            output_path = str(onnx_path)
            
            model_proto, _ = tf2onnx.convert.from_function(
                model.signatures["serving_default"],
                input_signature=spec,
                output_path=output_path
            )
            
            # Validate exported model
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            validation_result = self._validate_onnx_model_numpy(onnx_path, dummy_input)
            
            # Optimize if requested
            if optimize:
                optimized_path = self._optimize_onnx_model(onnx_path)
                onnx_path = optimized_path
            
            # Get model info
            model_info = self._get_model_info(onnx_path)
            
            # Upload to S3 if configured
            s3_path = None
            if self.s3_bucket:
                s3_path = self._upload_to_s3(onnx_path, model_name)
            
            result = {
                'model_name': model_name,
                'framework': 'tensorflow',
                'onnx_path': str(onnx_path),
                's3_path': s3_path,
                'input_shape': input_shape,
                'model_size_mb': onnx_path.stat().st_size / (1024 * 1024),
                'validation': validation_result,
                'model_info': model_info,
                'export_time': time.time()
            }
            
            logger.info(f"Successfully exported {model_name} to {onnx_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to export {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def export_huggingface_model(
        self,
        model_name_or_path: str,
        model_name: str,
        max_seq_length: int = 512,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """Export HuggingFace transformer model to ONNX"""
        logger.info(f"Exporting HuggingFace model: {model_name}")
        
        try:
            # Load model and tokenizer
            model = AutoModel.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Create dummy inputs
            dummy_text = "This is a sample text for model export validation."
            inputs = tokenizer(
                dummy_text,
                max_length=max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Set model to eval mode
            model.eval()
            
            # Export path
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # Dynamic axes for variable sequence length
            dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
            }
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (inputs['input_ids'], inputs['attention_mask']),
                    str(onnx_path),
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['last_hidden_state'],
                    dynamic_axes=dynamic_axes
                )
            
            # Validate exported model
            validation_result = self._validate_huggingface_model(
                onnx_path, model, tokenizer, dummy_text, max_seq_length
            )
            
            # Optimize if requested
            if optimize:
                optimized_path = self._optimize_onnx_model(onnx_path)
                onnx_path = optimized_path
            
            # Get model info
            model_info = self._get_model_info(onnx_path)
            
            # Upload to S3 if configured
            s3_path = None
            if self.s3_bucket:
                s3_path = self._upload_to_s3(onnx_path, model_name)
            
            result = {
                'model_name': model_name,
                'framework': 'huggingface',
                'onnx_path': str(onnx_path),
                's3_path': s3_path,
                'max_seq_length': max_seq_length,
                'model_size_mb': onnx_path.stat().st_size / (1024 * 1024),
                'validation': validation_result,
                'model_info': model_info,
                'export_time': time.time()
            }
            
            logger.info(f"Successfully exported {model_name} to {onnx_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to export {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _validate_onnx_model(
        self, 
        onnx_path: Path, 
        dummy_input: torch.Tensor, 
        original_model: nn.Module
    ) -> Dict[str, Any]:
        """Validate ONNX model against original PyTorch model"""
        try:
            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(str(onnx_path))
            
            # Run inference with ONNX Runtime
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Run inference with original model
            with torch.no_grad():
                original_model.eval()
                torch_outputs = original_model(dummy_input)
                if isinstance(torch_outputs, torch.Tensor):
                    torch_outputs = [torch_outputs]
                elif hasattr(torch_outputs, 'last_hidden_state'):
                    torch_outputs = [torch_outputs.last_hidden_state]
            
            # Compare outputs
            max_diff = 0.0
            for torch_out, ort_out in zip(torch_outputs, ort_outputs):
                diff = np.abs(torch_out.numpy() - ort_out).max()
                max_diff = max(max_diff, diff)
            
            validation_passed = max_diff < 1e-5
            
            return {
                'passed': validation_passed,
                'max_difference': float(max_diff),
                'onnx_check_passed': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'passed': False,
                'max_difference': None,
                'onnx_check_passed': False,
                'error': str(e)
            }
    
    def _validate_onnx_model_numpy(self, onnx_path: Path, dummy_input: np.ndarray) -> Dict[str, Any]:
        """Validate ONNX model with numpy input"""
        try:
            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(str(onnx_path))
            
            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            return {
                'passed': True,
                'onnx_check_passed': True,
                'output_shape': [out.shape for out in ort_outputs],
                'error': None
            }
            
        except Exception as e:
            return {
                'passed': False,
                'onnx_check_passed': False,
                'error': str(e)
            }
    
    def _validate_huggingface_model(
        self,
        onnx_path: Path,
        original_model: nn.Module,
        tokenizer,
        test_text: str,
        max_seq_length: int
    ) -> Dict[str, Any]:
        """Validate HuggingFace ONNX model"""
        try:
            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(str(onnx_path))
            
            # Prepare inputs
            inputs = tokenizer(
                test_text,
                max_length=max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Run ONNX inference
            ort_inputs = {
                'input_ids': inputs['input_ids'].numpy(),
                'attention_mask': inputs['attention_mask'].numpy()
            }
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Run original model inference
            with torch.no_grad():
                original_model.eval()
                torch_outputs = original_model(**inputs)
                torch_output = torch_outputs.last_hidden_state.numpy()
            
            # Compare outputs
            max_diff = np.abs(torch_output - ort_outputs[0]).max()
            validation_passed = max_diff < 1e-4
            
            return {
                'passed': validation_passed,
                'max_difference': float(max_diff),
                'onnx_check_passed': True,
                'output_shape': ort_outputs[0].shape,
                'error': None
            }
            
        except Exception as e:
            return {
                'passed': False,
                'max_difference': None,
                'onnx_check_passed': False,
                'error': str(e)
            }
    
    def _optimize_onnx_model(self, onnx_path: Path) -> Path:
        """Optimize ONNX model for inference"""
        try:
            from onnxruntime.tools import optimizer
            
            optimized_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"
            
            # Create optimization session
            opt_session = optimizer.optimize_model(
                str(onnx_path),
                model_type='bert',  # Works for most transformer models
                num_heads=0,  # Auto-detect
                hidden_size=0,  # Auto-detect
                optimization_level=99  # All optimizations
            )
            
            # Save optimized model
            opt_session.save_model_to_file(str(optimized_path))
            
            logger.info(f"Optimized model saved to {optimized_path}")
            return optimized_path
            
        except Exception as e:
            logger.warning(f"Failed to optimize model: {str(e)}")
            return onnx_path
    
    def _get_model_info(self, onnx_path: Path) -> Dict[str, Any]:
        """Get detailed model information"""
        try:
            model = onnx.load(str(onnx_path))
            
            # Get input/output info
            inputs = []
            for input_node in model.graph.input:
                shape = [dim.dim_value for dim in input_node.type.tensor_type.shape.dim]
                inputs.append({
                    'name': input_node.name,
                    'shape': shape,
                    'type': input_node.type.tensor_type.elem_type
                })
            
            outputs = []
            for output_node in model.graph.output:
                shape = [dim.dim_value for dim in output_node.type.tensor_type.shape.dim]
                outputs.append({
                    'name': output_node.name,
                    'shape': shape,
                    'type': output_node.type.tensor_type.elem_type
                })
            
            # Count parameters
            param_count = 0
            for init in model.graph.initializer:
                size = 1
                for dim in init.dims:
                    size *= dim
                param_count += size
            
            return {
                'inputs': inputs,
                'outputs': outputs,
                'parameter_count': param_count,
                'opset_version': model.opset_import[0].version if model.opset_import else None,
                'producer_name': model.producer_name,
                'model_version': model.model_version
            }
            
        except Exception as e:
            logger.warning(f"Failed to get model info: {str(e)}")
            return {}
    
    def _upload_to_s3(self, local_path: Path, model_name: str) -> Optional[str]:
        """Upload model to S3"""
        try:
            if not self.s3_client:
                return None
            
            s3_key = f"models/onnx/{model_name}/{local_path.name}"
            
            self.s3_client.upload_file(
                str(local_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'Metadata': {
                        'model_name': model_name,
                        'export_time': str(int(time.time())),
                        'framework': 'onnx'
                    }
                }
            )
            
            s3_path = f"s3://{self.s3_bucket}/{s3_key}"
            logger.info(f"Uploaded model to {s3_path}")
            return s3_path
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            return None

def create_sample_models() -> Dict[str, Any]:
    """Create sample models for testing"""
    models_info = {}
    
    logger.info("Creating sample models...")
    
    # ResNet-50 (Computer Vision)
    logger.info("Creating ResNet-50 model...")
    resnet = models.resnet50(pretrained=True)
    models_info['resnet50'] = {
        'model': resnet,
        'input_shape': (1, 3, 224, 224),
        'type': 'vision',
        'description': 'ResNet-50 for image classification'
    }
    
    # DistilBERT (NLP)
    logger.info("Creating DistilBERT model...")
    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    models_info['distilbert'] = {
        'model': distilbert,
        'tokenizer': DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
        'max_seq_length': 512,
        'type': 'nlp',
        'description': 'DistilBERT for text understanding'
    }
    
    # Simple CNN (Lightweight)
    logger.info("Creating simple CNN model...")
    simple_cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    models_info['simple_cnn'] = {
        'model': simple_cnn,
        'input_shape': (1, 3, 32, 32),
        'type': 'vision',
        'description': 'Simple CNN for lightweight inference'
    }
    
    return models_info

def export_all_sample_models(
    output_dir: str = "models/onnx",
    s3_bucket: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Export all sample models to ONNX"""
    
    exporter = ONNXExporter(output_dir, s3_bucket)
    sample_models = create_sample_models()
    results = []
    
    for model_name, model_info in sample_models.items():
        logger.info(f"Exporting {model_name}...")
        
        try:
            if model_info['type'] == 'nlp':
                # Export HuggingFace model
                # Use model name for HF models
                if model_name == 'distilbert':
                    result = exporter.export_huggingface_model(
                        'distilbert-base-uncased',
                        model_name,
                        model_info['max_seq_length']
                    )
                else:
                    result = exporter.export_pytorch_model(
                        model_info['model'],
                        model_name,
                        model_info['input_shape']
                    )
            else:
                # Export PyTorch model
                result = exporter.export_pytorch_model(
                    model_info['model'],
                    model_name,
                    model_info['input_shape']
                )
            
            result['description'] = model_info['description']
            result['type'] = model_info['type']
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to export {model_name}: {str(e)}")
            results.append({
                'model_name': model_name,
                'error': str(e),
                'success': False
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Export models to ONNX format')
    parser.add_argument('--output-dir', default='models/onnx', help='Output directory for ONNX models')
    parser.add_argument('--s3-bucket', help='S3 bucket for model upload')
    parser.add_argument('--model-name', help='Specific model to export')
    parser.add_argument('--model-path', help='Path to model file')
    parser.add_argument('--framework', choices=['pytorch', 'tensorflow', 'huggingface'], help='Model framework')
    parser.add_argument('--input-shape', help='Input shape (comma-separated)', default='1,3,224,224')
    parser.add_argument('--max-seq-length', type=int, default=512, help='Max sequence length for NLP models')
    parser.add_argument('--export-samples', action='store_true', help='Export sample models')
    parser.add_argument('--optimize', action='store_true', default=True, help='Optimize exported models')
    
    args = parser.parse_args()
    
    if args.export_samples:
        logger.info("Exporting sample models...")
        results = export_all_sample_models(args.output_dir, args.s3_bucket)
        
        # Save results
        results_file = Path(args.output_dir) / 'export_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Export results saved to {results_file}")
        
        # Print summary
        successful = sum(1 for r in results if r.get('validation', {}).get('passed', False))
        print(f"\n=== Export Summary ===")
        print(f"Total models: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        for result in results:
            status = "[PASS]" if result.get('validation', {}).get('passed', False) else "[FAIL]"
            size = result.get('model_size_mb', 0)
            print(f"{status} {result['model_name']}: {size:.1f} MB")
    
    elif args.model_name and args.model_path and args.framework:
        logger.info(f"Exporting custom model: {args.model_name}")
        exporter = ONNXExporter(args.output_dir, args.s3_bucket)
        
        input_shape = tuple(map(int, args.input_shape.split(',')))
        
        if args.framework == 'pytorch':
            # Load PyTorch model
            model = torch.load(args.model_path)
            result = exporter.export_pytorch_model(
                model, args.model_name, input_shape, optimize=args.optimize
            )
        elif args.framework == 'tensorflow':
            result = exporter.export_tensorflow_model(
                args.model_path, args.model_name, input_shape, optimize=args.optimize
            )
        elif args.framework == 'huggingface':
            result = exporter.export_huggingface_model(
                args.model_path, args.model_name, args.max_seq_length, optimize=args.optimize
            )
        
        print(f"Export completed: {result}")
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 