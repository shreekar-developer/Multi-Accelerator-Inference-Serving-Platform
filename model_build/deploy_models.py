#!/usr/bin/env python3
"""
Automated Model Deployment Pipeline
Handles complete model lifecycle: ONNX export → compilation → profiling → deployment
"""

import argparse
import json
import logging
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

import boto3
from botocore.exceptions import ClientError
import requests

from export_onnx import ONNXExporter, export_all_sample_models
from build_trt import TensorRTBuilder, build_engines_from_onnx_directory
from build_neuron import NeuronCompiler, compile_models_from_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDeploymentPipeline:
    """Orchestrates the complete model deployment pipeline"""
    
    def __init__(
        self,
        aws_region: str = "us-west-2",
        s3_bucket: Optional[str] = None,
        dynamodb_table: Optional[str] = None,
        ecr_registry: Optional[str] = None,
        k8s_namespace: str = "default"
    ):
        self.aws_region = aws_region
        self.s3_bucket = s3_bucket
        self.dynamodb_table = dynamodb_table
        self.ecr_registry = ecr_registry
        self.k8s_namespace = k8s_namespace
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=aws_region) if s3_bucket else None
        self.dynamodb_client = boto3.client('dynamodb', region_name=aws_region) if dynamodb_table else None
        self.ecr_client = boto3.client('ecr', region_name=aws_region) if ecr_registry else None
        
        # Create output directories
        self.work_dir = Path("deployment_work")
        self.work_dir.mkdir(exist_ok=True)
        
        self.onnx_dir = self.work_dir / "onnx"
        self.tensorrt_dir = self.work_dir / "tensorrt"
        self.neuron_dir = self.work_dir / "neuron"
        self.profiles_dir = self.work_dir / "profiles"
        
        for dir_path in [self.onnx_dir, self.tensorrt_dir, self.neuron_dir, self.profiles_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def deploy_complete_pipeline(
        self,
        model_config: Dict[str, Any],
        deploy_targets: List[str] = ["cpu", "gpu", "inferentia"],
        canary_enabled: bool = True,
        auto_rollback: bool = True
    ) -> Dict[str, Any]:
        """Run complete deployment pipeline for a model"""
        model_name = model_config["name"]
        model_version = model_config.get("version", "v1.0")
        
        logger.info(f"Starting deployment pipeline for {model_name} v{model_version}")
        
        deployment_result = {
            'model_name': model_name,
            'model_version': model_version,
            'pipeline_start_time': time.time(),
            'stages': {},
            'success': False,
            'error': None
        }
        
        try:
            # Stage 1: Export to ONNX
            logger.info("Stage 1: Exporting to ONNX...")
            onnx_result = self._export_model_to_onnx(model_config)
            deployment_result['stages']['onnx_export'] = onnx_result
            
            if not onnx_result.get('success', False):
                raise Exception(f"ONNX export failed: {onnx_result.get('error')}")
            
            # Stage 2: Compile for target accelerators
            logger.info("Stage 2: Compiling for target accelerators...")
            compilation_results = self._compile_for_targets(
                onnx_result['onnx_path'], model_name, deploy_targets
            )
            deployment_result['stages']['compilation'] = compilation_results
            
            # Stage 3: Profile models
            logger.info("Stage 3: Profiling compiled models...")
            profiling_results = self._profile_models(compilation_results, model_config)
            deployment_result['stages']['profiling'] = profiling_results
            
            # Stage 4: Store model profiles
            logger.info("Stage 4: Storing model profiles...")
            profile_storage_result = self._store_model_profiles(profiling_results, model_name, model_version)
            deployment_result['stages']['profile_storage'] = profile_storage_result
            
            # Stage 5: Deploy to Kubernetes
            logger.info("Stage 5: Deploying to Kubernetes...")
            k8s_deployment_result = self._deploy_to_kubernetes(
                model_config, compilation_results, canary_enabled
            )
            deployment_result['stages']['k8s_deployment'] = k8s_deployment_result
            
            # Stage 6: Validate deployment
            logger.info("Stage 6: Validating deployment...")
            validation_result = self._validate_deployment(model_name, model_version, deploy_targets)
            deployment_result['stages']['validation'] = validation_result
            
            # Stage 7: Canary promotion or rollback
            if canary_enabled:
                logger.info("Stage 7: Managing canary deployment...")
                canary_result = self._manage_canary_deployment(
                    model_name, model_version, validation_result, auto_rollback
                )
                deployment_result['stages']['canary_management'] = canary_result
            
            deployment_result['success'] = True
            deployment_result['pipeline_end_time'] = time.time()
            deployment_result['total_duration_seconds'] = (
                deployment_result['pipeline_end_time'] - deployment_result['pipeline_start_time']
            )
            
            logger.info(f"Deployment pipeline completed successfully for {model_name} v{model_version}")
            return deployment_result
            
        except Exception as e:
            error_msg = f"Deployment pipeline failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            deployment_result['error'] = error_msg
            deployment_result['pipeline_end_time'] = time.time()
            
            # Attempt rollback if deployment was partially successful
            if canary_enabled and auto_rollback:
                logger.info("Attempting automatic rollback...")
                try:
                    rollback_result = self._rollback_deployment(model_name, model_version)
                    deployment_result['rollback'] = rollback_result
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {str(rollback_error)}")
            
            return deployment_result
    
    def _export_model_to_onnx(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Export model to ONNX format"""
        try:
            model_name = model_config["name"]
            model_source = model_config.get("source", {})
            
            exporter = ONNXExporter(str(self.onnx_dir), self.s3_bucket)
            
            if model_source.get("type") == "huggingface":
                result = exporter.export_huggingface_model(
                    model_source["path"],
                    model_name,
                    model_config.get("max_seq_length", 512)
                )
            elif model_source.get("type") == "pytorch":
                # Load PyTorch model
                import torch
                model = torch.load(model_source["path"])
                input_shape = tuple(model_config.get("input_shape", [1, 3, 224, 224]))
                result = exporter.export_pytorch_model(model, model_name, input_shape)
            elif model_source.get("type") == "sample":
                # Export sample models
                results = export_all_sample_models(str(self.onnx_dir), self.s3_bucket)
                # Find the specific model
                result = next((r for r in results if r["model_name"] == model_name), None)
                if not result:
                    raise ValueError(f"Sample model {model_name} not found")
            else:
                raise ValueError(f"Unsupported model source type: {model_source.get('type')}")
            
            return {
                'success': True,
                'onnx_path': result['onnx_path'],
                'model_size_mb': result.get('model_size_mb', 0),
                'validation': result.get('validation', {}),
                's3_path': result.get('s3_path')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _compile_for_targets(
        self,
        onnx_path: str,
        model_name: str,
        targets: List[str]
    ) -> Dict[str, Any]:
        """Compile model for target accelerators"""
        compilation_results = {}
        
        for target in targets:
            try:
                if target == "gpu":
                    logger.info(f"Compiling {model_name} for GPU (TensorRT)...")
                    builder = TensorRTBuilder(str(self.tensorrt_dir), self.s3_bucket)
                    results = builder.build_multiple_precisions(
                        onnx_path=onnx_path,
                        engine_name=model_name,
                        precisions=["fp16", "fp32"],
                        max_batch_sizes=[1, 8, 32]
                    )
                    compilation_results[target] = {
                        'success': True,
                        'engines': results,
                        'framework': 'tensorrt'
                    }
                
                elif target == "inferentia":
                    logger.info(f"Compiling {model_name} for Inferentia (Neuron)...")
                    compiler = NeuronCompiler(str(self.neuron_dir), self.s3_bucket)
                    result = compiler.compile_onnx_model(
                        onnx_path=onnx_path,
                        model_name=model_name,
                        input_shape=(1, 3, 224, 224),  # Default shape
                        optimization_level="2"
                    )
                    compilation_results[target] = {
                        'success': True,
                        'compiled_model': result,
                        'framework': 'neuron'
                    }
                
                elif target == "cpu":
                    # CPU uses ONNX directly, no additional compilation needed
                    compilation_results[target] = {
                        'success': True,
                        'onnx_path': onnx_path,
                        'framework': 'onnx_runtime'
                    }
                
                else:
                    logger.warning(f"Unknown target: {target}")
                    compilation_results[target] = {
                        'success': False,
                        'error': f"Unknown target: {target}"
                    }
                    
            except Exception as e:
                logger.error(f"Compilation failed for {target}: {str(e)}")
                compilation_results[target] = {
                    'success': False,
                    'error': str(e)
                }
        
        return compilation_results
    
    def _profile_models(
        self,
        compilation_results: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Profile compiled models for performance characteristics"""
        profiling_results = {}
        
        for target, result in compilation_results.items():
            if not result.get('success', False):
                continue
            
            try:
                logger.info(f"Profiling {target} model...")
                
                if target == "gpu" and result.get('framework') == 'tensorrt':
                    # Profile TensorRT engines
                    profiles = []
                    for engine_result in result.get('engines', []):
                        if engine_result.get('validation', {}).get('passed', False):
                            profile = self._create_model_profile(
                                model_config, target, engine_result
                            )
                            profiles.append(profile)
                    
                    profiling_results[target] = {
                        'success': True,
                        'profiles': profiles
                    }
                
                elif target == "inferentia" and result.get('framework') == 'neuron':
                    # Profile Neuron model
                    profile = self._create_model_profile(
                        model_config, target, result['compiled_model']
                    )
                    profiling_results[target] = {
                        'success': True,
                        'profiles': [profile]
                    }
                
                elif target == "cpu":
                    # Create CPU profile (simulated)
                    profile = self._create_cpu_profile(model_config)
                    profiling_results[target] = {
                        'success': True,
                        'profiles': [profile]
                    }
                
            except Exception as e:
                logger.error(f"Profiling failed for {target}: {str(e)}")
                profiling_results[target] = {
                    'success': False,
                    'error': str(e)
                }
        
        return profiling_results
    
    def _create_model_profile(
        self,
        model_config: Dict[str, Any],
        accelerator: str,
        compilation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a model profile entry"""
        
        # Simulate benchmark results based on accelerator type and model
        if accelerator == "gpu":
            base_latency = 15.0  # GPU base latency
            base_qps = 200.0
        elif accelerator == "inferentia":
            base_latency = 25.0  # Inferentia base latency
            base_qps = 150.0
        else:  # CPU
            base_latency = 80.0  # CPU base latency
            base_qps = 50.0
        
        # Adjust based on batch size
        batch_size = compilation_result.get('max_batch_size', 1)
        adjusted_latency = base_latency * (1 + (batch_size - 1) * 0.3)
        adjusted_qps = base_qps / batch_size
        
        profile = {
            'model_id': model_config['name'],
            'version': model_config.get('version', 'v1.0'),
            'accelerator': accelerator,
            'batch_size': batch_size,
            'sequence_length': model_config.get('max_seq_length', 512),
            'p50_latency_ms': adjusted_latency * 0.8,
            'p95_latency_ms': adjusted_latency,
            'p99_latency_ms': adjusted_latency * 1.2,
            'qps_sustained': adjusted_qps,
            'memory_mb': compilation_result.get('model_size_mb', 100) * 10,
            'cpu_utilization': 70.0 if accelerator == "cpu" else 30.0,
            'gpu_utilization': 85.0 if accelerator == "gpu" else 0.0,
            'precision': compilation_result.get('precision', 'fp32'),
            'artifact_path': compilation_result.get('engine_path') or compilation_result.get('model_path'),
            's3_path': compilation_result.get('s3_path'),
            'cold_start_ms': 1000.0 if accelerator == "inferentia" else 500.0,
            'warmup_requests': 5,
            'max_concurrency': 32 if accelerator == "gpu" else 16,
            'last_updated': time.time(),
            'profile_version': 1,
            'benchmark_config': {
                'test_duration': 60.0,
                'concurrency_levels': [1, 4, 8, 16],
                'batch_sizes': [1, 8, 16, 32],
                'environment': {
                    'accelerator': accelerator,
                    'framework': compilation_result.get('framework', 'unknown')
                }
            }
        }
        
        return profile
    
    def _create_cpu_profile(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create CPU-specific profile"""
        return self._create_model_profile(
            model_config,
            "cpu",
            {
                'max_batch_size': 1,
                'model_size_mb': 50,
                'framework': 'onnx_runtime'
            }
        )
    
    def _store_model_profiles(
        self,
        profiling_results: Dict[str, Any],
        model_name: str,
        model_version: str
    ) -> Dict[str, Any]:
        """Store model profiles in DynamoDB"""
        if not self.dynamodb_client or not self.dynamodb_table:
            logger.warning("DynamoDB not configured, skipping profile storage")
            return {'success': False, 'error': 'DynamoDB not configured'}
        
        try:
            stored_profiles = []
            
            for target, result in profiling_results.items():
                if not result.get('success', False):
                    continue
                
                for profile in result.get('profiles', []):
                    # Store profile in DynamoDB
                    pk = f"{model_name}#{model_version}"
                    sk = f"{target}#{profile['batch_size']}#{profile['sequence_length']}"
                    
                    item = {
                        'model_version': {'S': pk},
                        'accelerator_config': {'S': sk},
                        'model_id': {'S': profile['model_id']},
                        'version': {'S': profile['version']},
                        'accelerator': {'S': profile['accelerator']},
                        'p50_latency_ms': {'N': str(profile['p50_latency_ms'])},
                        'p95_latency_ms': {'N': str(profile['p95_latency_ms'])},
                        'p99_latency_ms': {'N': str(profile['p99_latency_ms'])},
                        'qps_sustained': {'N': str(profile['qps_sustained'])},
                        'memory_mb': {'N': str(profile['memory_mb'])},
                        'last_updated': {'S': str(int(profile['last_updated']))}
                    }
                    
                    self.dynamodb_client.put_item(
                        TableName=self.dynamodb_table,
                        Item=item
                    )
                    
                    stored_profiles.append({
                        'model_name': model_name,
                        'accelerator': target,
                        'batch_size': profile['batch_size'],
                        'profile_key': f"{pk}#{sk}"
                    })
            
            return {
                'success': True,
                'stored_profiles': stored_profiles,
                'total_profiles': len(stored_profiles)
            }
            
        except Exception as e:
            logger.error(f"Failed to store profiles: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _deploy_to_kubernetes(
        self,
        model_config: Dict[str, Any],
        compilation_results: Dict[str, Any],
        canary_enabled: bool = True
    ) -> Dict[str, Any]:
        """Deploy models to Kubernetes"""
        try:
            model_name = model_config['name']
            model_version = model_config.get('version', 'v1.0')
            
            # Generate Kubernetes manifests
            manifests = self._generate_k8s_manifests(model_config, compilation_results, canary_enabled)
            
            # Apply manifests
            deployment_results = []
            for manifest in manifests:
                result = self._apply_k8s_manifest(manifest)
                deployment_results.append(result)
            
            # Wait for deployments to be ready
            ready_deployments = self._wait_for_deployments(model_name, model_version)
            
            return {
                'success': True,
                'manifests_applied': len(manifests),
                'deployment_results': deployment_results,
                'ready_deployments': ready_deployments,
                'canary_enabled': canary_enabled
            }
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_k8s_manifests(
        self,
        model_config: Dict[str, Any],
        compilation_results: Dict[str, Any],
        canary_enabled: bool
    ) -> List[Dict[str, Any]]:
        """Generate Kubernetes deployment manifests"""
        manifests = []
        model_name = model_config['name']
        model_version = model_config.get('version', 'v1.0')
        
        for target, result in compilation_results.items():
            if not result.get('success', False):
                continue
            
            # Create deployment manifest
            deployment = {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': f"{model_name}-{target}-{'canary' if canary_enabled else 'stable'}",
                    'namespace': self.k8s_namespace,
                    'labels': {
                        'app': f"{model_name}-{target}",
                        'model': model_name,
                        'version': model_version,
                        'accelerator': target,
                        'deployment-type': 'canary' if canary_enabled else 'stable'
                    }
                },
                'spec': {
                    'replicas': 1 if canary_enabled else 3,
                    'selector': {
                        'matchLabels': {
                            'app': f"{model_name}-{target}",
                            'model': model_name,
                            'version': model_version
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': f"{model_name}-{target}",
                                'model': model_name,
                                'version': model_version,
                                'accelerator': target
                            }
                        },
                        'spec': {
                            'containers': [{
                                'name': 'model-server',
                                'image': f"{self.ecr_registry}/{target}-backend:latest",
                                'ports': [{'containerPort': 8080}],
                                'env': [
                                    {'name': 'MODEL_NAME', 'value': model_name},
                                    {'name': 'MODEL_VERSION', 'value': model_version},
                                    {'name': 'ACCELERATOR', 'value': target}
                                ],
                                'resources': self._get_resource_requirements(target),
                                'livenessProbe': {
                                    'httpGet': {'path': '/health', 'port': 8080},
                                    'initialDelaySeconds': 30,
                                    'periodSeconds': 30
                                },
                                'readinessProbe': {
                                    'httpGet': {'path': '/health', 'port': 8080},
                                    'initialDelaySeconds': 10,
                                    'periodSeconds': 10
                                }
                            }],
                            'nodeSelector': {'accelerator': target},
                            'tolerations': [{
                                'key': 'accelerator',
                                'operator': 'Equal',
                                'value': target,
                                'effect': 'NoSchedule'
                            }]
                        }
                    }
                }
            }
            
            manifests.append(deployment)
            
            # Create service if it doesn't exist
            service = {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': f"{model_name}-{target}-service",
                    'namespace': self.k8s_namespace,
                    'labels': {
                        'app': f"{model_name}-{target}",
                        'model': model_name
                    }
                },
                'spec': {
                    'type': 'ClusterIP',
                    'ports': [{'port': 8080, 'targetPort': 8080}],
                    'selector': {
                        'app': f"{model_name}-{target}",
                        'model': model_name
                    }
                }
            }
            
            manifests.append(service)
        
        return manifests
    
    def _get_resource_requirements(self, accelerator: str) -> Dict[str, Any]:
        """Get resource requirements for accelerator type"""
        if accelerator == "gpu":
            return {
                'requests': {'nvidia.com/gpu': 1, 'memory': '4Gi', 'cpu': '2'},
                'limits': {'nvidia.com/gpu': 1, 'memory': '8Gi', 'cpu': '4'}
            }
        elif accelerator == "inferentia":
            return {
                'requests': {'aws.amazon.com/neuron': 1, 'memory': '2Gi', 'cpu': '2'},
                'limits': {'aws.amazon.com/neuron': 1, 'memory': '4Gi', 'cpu': '4'}
            }
        else:  # CPU
            return {
                'requests': {'memory': '1Gi', 'cpu': '1'},
                'limits': {'memory': '2Gi', 'cpu': '2'}
            }
    
    def _apply_k8s_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Kubernetes manifest"""
        try:
            # Write manifest to file
            manifest_file = self.work_dir / f"manifest_{manifest['metadata']['name']}.yaml"
            with open(manifest_file, 'w') as f:
                yaml.dump(manifest, f)
            
            # Apply with kubectl
            result = subprocess.run(
                ['kubectl', 'apply', '-f', str(manifest_file)],
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                'success': True,
                'manifest_name': manifest['metadata']['name'],
                'output': result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'manifest_name': manifest['metadata']['name'],
                'error': e.stderr
            }
    
    def _wait_for_deployments(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Wait for deployments to be ready"""
        try:
            # Wait for deployments to be available
            result = subprocess.run(
                [
                    'kubectl', 'wait', '--for=condition=available',
                    f'deployment/{model_name}-cpu-canary',
                    f'deployment/{model_name}-gpu-canary',
                    f'deployment/{model_name}-neuron-canary',
                    '--timeout=300s',
                    f'--namespace={self.k8s_namespace}'
                ],
                capture_output=True,
                text=True,
                timeout=320
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Timeout waiting for deployments to be ready'
            }
    
    def _validate_deployment(
        self,
        model_name: str,
        model_version: str,
        targets: List[str]
    ) -> Dict[str, Any]:
        """Validate deployed models"""
        validation_results = {}
        
        for target in targets:
            try:
                # Test endpoint
                service_url = f"http://{model_name}-{target}-service.{self.k8s_namespace}.svc.cluster.local:8080"
                
                # Health check
                health_response = requests.get(f"{service_url}/health", timeout=10)
                
                # Prediction test
                test_request = {
                    "model_id": model_name,
                    "version": model_version,
                    "inputs": [{"data": [1, 2, 3, 4, 5]}]
                }
                
                predict_response = requests.post(
                    f"{service_url}/predict",
                    json=test_request,
                    timeout=30
                )
                
                validation_results[target] = {
                    'success': health_response.status_code == 200 and predict_response.status_code == 200,
                    'health_status': health_response.status_code,
                    'prediction_status': predict_response.status_code,
                    'prediction_latency_ms': predict_response.elapsed.total_seconds() * 1000,
                    'response_data': predict_response.json() if predict_response.status_code == 200 else None
                }
                
            except Exception as e:
                validation_results[target] = {
                    'success': False,
                    'error': str(e)
                }
        
        return validation_results
    
    def _manage_canary_deployment(
        self,
        model_name: str,
        model_version: str,
        validation_result: Dict[str, Any],
        auto_rollback: bool = True
    ) -> Dict[str, Any]:
        """Manage canary deployment promotion or rollback"""
        try:
            # Check if all validations passed
            all_passed = all(
                result.get('success', False) 
                for result in validation_result.values()
            )
            
            if all_passed:
                # Promote canary to production
                logger.info("All validations passed, promoting canary to production")
                promotion_result = self._promote_canary(model_name, model_version)
                
                return {
                    'action': 'promoted',
                    'success': promotion_result.get('success', False),
                    'details': promotion_result
                }
            
            elif auto_rollback:
                # Rollback canary deployment
                logger.warning("Validation failed, rolling back canary deployment")
                rollback_result = self._rollback_deployment(model_name, model_version)
                
                return {
                    'action': 'rolled_back',
                    'success': rollback_result.get('success', False),
                    'details': rollback_result
                }
            
            else:
                return {
                    'action': 'manual_intervention_required',
                    'success': False,
                    'validation_failures': validation_result
                }
                
        except Exception as e:
            logger.error(f"Canary management failed: {str(e)}")
            return {
                'action': 'error',
                'success': False,
                'error': str(e)
            }
    
    def _promote_canary(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Promote canary deployment to production"""
        try:
            # Scale up canary deployments to production levels
            targets = ["cpu", "gpu", "neuron"]
            promotion_results = []
            
            for target in targets:
                result = subprocess.run(
                    [
                        'kubectl', 'scale', 
                        f'deployment/{model_name}-{target}-canary',
                        '--replicas=3',
                        f'--namespace={self.k8s_namespace}'
                    ],
                    capture_output=True,
                    text=True
                )
                
                promotion_results.append({
                    'target': target,
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr if result.returncode != 0 else None
                })
            
            # Update Istio routing to send 100% traffic to canary
            istio_result = self._update_istio_routing(model_name, model_version, canary_weight=100)
            
            return {
                'success': all(r['success'] for r in promotion_results),
                'scaling_results': promotion_results,
                'istio_routing': istio_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _rollback_deployment(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Rollback failed deployment"""
        try:
            # Delete canary deployments
            targets = ["cpu", "gpu", "neuron"]
            rollback_results = []
            
            for target in targets:
                result = subprocess.run(
                    [
                        'kubectl', 'delete', 
                        f'deployment/{model_name}-{target}-canary',
                        f'--namespace={self.k8s_namespace}',
                        '--ignore-not-found=true'
                    ],
                    capture_output=True,
                    text=True
                )
                
                rollback_results.append({
                    'target': target,
                    'success': result.returncode == 0,
                    'output': result.stdout
                })
            
            # Reset Istio routing to stable version
            istio_result = self._update_istio_routing(model_name, model_version, canary_weight=0)
            
            return {
                'success': all(r['success'] for r in rollback_results),
                'deletion_results': rollback_results,
                'istio_routing': istio_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_istio_routing(
        self,
        model_name: str,
        model_version: str,
        canary_weight: int = 10
    ) -> Dict[str, Any]:
        """Update Istio routing for canary deployment"""
        try:
            # This is a placeholder for Istio VirtualService update
            # In practice, you'd update the VirtualService to split traffic
            
            virtual_service = {
                'apiVersion': 'networking.istio.io/v1alpha3',
                'kind': 'VirtualService',
                'metadata': {
                    'name': f"{model_name}-routing",
                    'namespace': self.k8s_namespace
                },
                'spec': {
                    'http': [{
                        'match': [{'uri': {'prefix': f'/{model_name}'}}],
                        'route': [
                            {
                                'destination': {'host': f"{model_name}-service"},
                                'weight': 100 - canary_weight
                            },
                            {
                                'destination': {'host': f"{model_name}-canary-service"},
                                'weight': canary_weight
                            }
                        ]
                    }]
                }
            }
            
            # Apply VirtualService
            vs_file = self.work_dir / f"virtualservice_{model_name}.yaml"
            with open(vs_file, 'w') as f:
                yaml.dump(virtual_service, f)
            
            result = subprocess.run(
                ['kubectl', 'apply', '-f', str(vs_file)],
                capture_output=True,
                text=True
            )
            
            return {
                'success': result.returncode == 0,
                'canary_weight': canary_weight,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_sample_model_config() -> Dict[str, Any]:
    """Create a sample model configuration"""
    return {
        'name': 'resnet50',
        'version': 'v1.0',
        'source': {
            'type': 'sample'
        },
        'input_shape': [1, 3, 224, 224],
        'sla_requirements': {
            'gold_latency_ms': 50,
            'silver_latency_ms': 150
        },
        'deployment': {
            'targets': ['cpu', 'gpu', 'inferentia'],
            'canary_enabled': True,
            'auto_rollback': True
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Deploy models through complete pipeline')
    parser.add_argument('--config', help='Model configuration YAML file')
    parser.add_argument('--model-name', help='Model name for single model deployment')
    parser.add_argument('--model-version', default='v1.0', help='Model version')
    parser.add_argument('--targets', nargs='+', default=['cpu', 'gpu', 'inferentia'], 
                       choices=['cpu', 'gpu', 'inferentia'], help='Deployment targets')
    parser.add_argument('--aws-region', default='us-west-2', help='AWS region')
    parser.add_argument('--s3-bucket', help='S3 bucket for model storage')
    parser.add_argument('--dynamodb-table', help='DynamoDB table for model profiles')
    parser.add_argument('--ecr-registry', help='ECR registry for container images')
    parser.add_argument('--k8s-namespace', default='default', help='Kubernetes namespace')
    parser.add_argument('--no-canary', action='store_true', help='Disable canary deployment')
    parser.add_argument('--no-auto-rollback', action='store_true', help='Disable auto rollback')
    parser.add_argument('--sample-models', action='store_true', help='Deploy sample models')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ModelDeploymentPipeline(
        aws_region=args.aws_region,
        s3_bucket=args.s3_bucket,
        dynamodb_table=args.dynamodb_table,
        ecr_registry=args.ecr_registry,
        k8s_namespace=args.k8s_namespace
    )
    
    if args.sample_models:
        logger.info("Deploying sample models...")
        sample_models = ['resnet50', 'distilbert', 'simple_cnn']
        
        all_results = []
        for model_name in sample_models:
            model_config = create_sample_model_config()
            model_config['name'] = model_name
            
            result = pipeline.deploy_complete_pipeline(
                model_config=model_config,
                deploy_targets=args.targets,
                canary_enabled=not args.no_canary,
                auto_rollback=not args.no_auto_rollback
            )
            
            all_results.append(result)
        
        # Save results
        results_file = Path("deployment_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Deployment results saved to {results_file}")
        
        # Print summary
        successful = sum(1 for r in all_results if r.get('success', False))
        print(f"\n=== Deployment Summary ===")
        print(f"Total models: {len(all_results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(all_results) - successful}")
        
        for result in all_results:
            status = "[PASS]" if result.get('success', False) else "[FAIL]"
            duration = result.get('total_duration_seconds', 0)
            print(f"{status} {result['model_name']} v{result['model_version']}: {duration:.1f}s")
    
    elif args.config:
        logger.info(f"Deploying model from config: {args.config}")
        model_config = load_model_config(args.config)
        
        result = pipeline.deploy_complete_pipeline(
            model_config=model_config,
            deploy_targets=args.targets,
            canary_enabled=not args.no_canary,
            auto_rollback=not args.no_auto_rollback
        )
        
        print(f"Deployment result: {json.dumps(result, indent=2, default=str)}")
    
    elif args.model_name:
        logger.info(f"Deploying single model: {args.model_name}")
        model_config = create_sample_model_config()
        model_config['name'] = args.model_name
        model_config['version'] = args.model_version
        
        result = pipeline.deploy_complete_pipeline(
            model_config=model_config,
            deploy_targets=args.targets,
            canary_enabled=not args.no_canary,
            auto_rollback=not args.no_auto_rollback
        )
        
        print(f"Deployment result: {json.dumps(result, indent=2, default=str)}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 