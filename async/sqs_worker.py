#!/usr/bin/env python3
"""
SQS Async Worker for Batch Inference Processing
Processes batch inference requests from SQS queue and routes to appropriate backends.
"""

import json
import boto3
import asyncio
import aiohttp
import logging
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Batch inference request structure"""
    request_id: str
    model_id: str
    version: str
    sla_tier: str
    inputs: List[Dict[str, Any]]
    callback_url: Optional[str] = None
    priority: int = 0
    submitted_at: str = ""
    max_batch_size: int = 32
    timeout_seconds: int = 300

@dataclass
class BatchResponse:
    """Batch inference response structure"""
    request_id: str
    status: str  # "completed", "failed", "partial"
    outputs: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processed_at: str
    latency_ms: float
    accelerator_used: str
    cost_estimate: float

class SQSWorker:
    """SQS Worker for processing batch inference requests"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sqs = boto3.client('sqs', region_name=config.get('aws_region', 'us-west-2'))
        self.s3 = boto3.client('s3', region_name=config.get('aws_region', 'us-west-2'))
        self.dynamodb = boto3.client('dynamodb', region_name=config.get('aws_region', 'us-west-2'))
        
        self.queue_url = config['sqs_queue_url']
        self.results_bucket = config.get('results_bucket', 'inference-batch-results')
        self.router_endpoint = config.get('router_endpoint', 'http://router:8080')
        self.max_workers = config.get('max_workers', 10)
        self.batch_timeout = config.get('batch_timeout', 300)
        self.max_batch_size = config.get('max_batch_size', 32)
        self.poll_interval = config.get('poll_interval', 5)
        
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.running = True
        self.active_batches = {}
        self.batch_accumulator = {}
        self.metrics = {
            'requests_processed': 0,
            'batches_completed': 0,
            'errors': 0,
            'total_latency_ms': 0,
            'start_time': time.time()
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def start(self):
        """Start the SQS worker"""
        logger.info("Starting SQS worker...")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._poll_sqs()),
            asyncio.create_task(self._process_batches()),
            asyncio.create_task(self._cleanup_expired_batches()),
            asyncio.create_task(self._report_metrics())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            await self._shutdown()
    
    async def _poll_sqs(self):
        """Poll SQS for new messages"""
        while self.running:
            try:
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=10,
                    WaitTimeSeconds=20,  # Long polling
                    MessageAttributeNames=['All']
                )
                
                messages = response.get('Messages', [])
                if messages:
                    logger.info(f"Received {len(messages)} messages from SQS")
                    await self._process_messages(messages)
                
            except Exception as e:
                logger.error(f"Error polling SQS: {e}")
                await asyncio.sleep(self.poll_interval)
    
    async def _process_messages(self, messages: List[Dict]):
        """Process received SQS messages"""
        for message in messages:
            try:
                # Parse message body
                body = json.loads(message['Body'])
                batch_request = BatchRequest(**body)
                
                # Add to batch accumulator
                await self._add_to_batch(batch_request)
                
                # Delete message from queue
                self.sqs.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # Message will be retried automatically
    
    async def _add_to_batch(self, request: BatchRequest):
        """Add request to batch accumulator for optimal batching"""
        batch_key = f"{request.model_id}#{request.version}#{request.sla_tier}"
        
        if batch_key not in self.batch_accumulator:
            self.batch_accumulator[batch_key] = {
                'requests': [],
                'created_at': time.time(),
                'model_id': request.model_id,
                'version': request.version,
                'sla_tier': request.sla_tier
            }
        
        self.batch_accumulator[batch_key]['requests'].append(request)
        
        # Check if batch is ready for processing
        batch_info = self.batch_accumulator[batch_key]
        should_process = (
            len(batch_info['requests']) >= self.max_batch_size or
            time.time() - batch_info['created_at'] > 30 or  # 30 second timeout
            request.sla_tier == 'gold'  # Process gold tier immediately
        )
        
        if should_process:
            await self._submit_batch(batch_key)
    
    async def _submit_batch(self, batch_key: str):
        """Submit accumulated batch for processing"""
        if batch_key not in self.batch_accumulator:
            return
        
        batch_info = self.batch_accumulator.pop(batch_key)
        requests = batch_info['requests']
        
        if not requests:
            return
        
        batch_id = str(uuid.uuid4())
        logger.info(f"Submitting batch {batch_id} with {len(requests)} requests")
        
        # Create batch processing task
        self.active_batches[batch_id] = {
            'requests': requests,
            'started_at': time.time(),
            'model_id': batch_info['model_id'],
            'version': batch_info['version'],
            'sla_tier': batch_info['sla_tier']
        }
        
        # Process batch asynchronously
        asyncio.create_task(self._process_batch(batch_id))
    
    async def _process_batch(self, batch_id: str):
        """Process a batch of inference requests"""
        if batch_id not in self.active_batches:
            return
        
        batch_info = self.active_batches[batch_id]
        requests = batch_info['requests']
        start_time = time.time()
        
        try:
            # Combine all inputs for batch processing
            combined_inputs = []
            request_mapping = []
            
            for i, req in enumerate(requests):
                for j, input_data in enumerate(req.inputs):
                    combined_inputs.append(input_data)
                    request_mapping.append((i, j))  # (request_index, input_index)
            
            # Create batch inference request
            batch_payload = {
                'model_id': batch_info['model_id'],
                'version': batch_info['version'],
                'sla_tier': 'bronze',  # Batch requests are typically bronze tier
                'inputs': combined_inputs,
                'batch_mode': True,
                'batch_id': batch_id
            }
            
            # Send to router
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.router_endpoint}/predict",
                    json=batch_payload,
                    timeout=aiohttp.ClientTimeout(total=self.batch_timeout)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        await self._handle_batch_success(batch_id, result, request_mapping)
                    else:
                        error_text = await response.text()
                        await self._handle_batch_error(batch_id, f"HTTP {response.status}: {error_text}")
            
        except asyncio.TimeoutError:
            await self._handle_batch_error(batch_id, "Batch processing timeout")
        except Exception as e:
            await self._handle_batch_error(batch_id, str(e))
        finally:
            # Clean up
            if batch_id in self.active_batches:
                processing_time = time.time() - start_time
                self.metrics['total_latency_ms'] += processing_time * 1000
                del self.active_batches[batch_id]
    
    async def _handle_batch_success(self, batch_id: str, result: Dict, request_mapping: List[tuple]):
        """Handle successful batch processing"""
        batch_info = self.active_batches[batch_id]
        requests = batch_info['requests']
        outputs = result.get('outputs', [])
        
        # Map outputs back to original requests
        request_outputs = [[] for _ in requests]
        for output_idx, (req_idx, input_idx) in enumerate(request_mapping):
            if output_idx < len(outputs):
                request_outputs[req_idx].append(outputs[output_idx])
        
        # Create individual responses
        responses = []
        for i, req in enumerate(requests):
            response = BatchResponse(
                request_id=req.request_id,
                status="completed",
                outputs=request_outputs[i],
                metadata={
                    'batch_id': batch_id,
                    'batch_size': len(requests),
                    'model_id': req.model_id,
                    'version': req.version
                },
                processed_at=datetime.utcnow().isoformat(),
                latency_ms=result.get('latency_ms', 0),
                accelerator_used=result.get('accelerator', 'unknown'),
                cost_estimate=result.get('cost_estimate', 0.0)
            )
            responses.append(response)
        
        # Store results and send callbacks
        await self._store_results(batch_id, responses)
        await self._send_callbacks(responses)
        
        # Update metrics
        self.metrics['requests_processed'] += len(requests)
        self.metrics['batches_completed'] += 1
        
        logger.info(f"Batch {batch_id} completed successfully with {len(requests)} requests")
    
    async def _handle_batch_error(self, batch_id: str, error_message: str):
        """Handle batch processing errors"""
        batch_info = self.active_batches[batch_id]
        requests = batch_info['requests']
        
        # Create error responses
        responses = []
        for req in requests:
            response = BatchResponse(
                request_id=req.request_id,
                status="failed",
                outputs=[],
                metadata={
                    'batch_id': batch_id,
                    'error': error_message,
                    'model_id': req.model_id,
                    'version': req.version
                },
                processed_at=datetime.utcnow().isoformat(),
                latency_ms=0,
                accelerator_used='none',
                cost_estimate=0.0
            )
            responses.append(response)
        
        # Store error results
        await self._store_results(batch_id, responses)
        await self._send_callbacks(responses)
        
        # Update metrics
        self.metrics['errors'] += len(requests)
        
        logger.error(f"Batch {batch_id} failed: {error_message}")
    
    async def _store_results(self, batch_id: str, responses: List[BatchResponse]):
        """Store batch results in S3"""
        try:
            results_data = {
                'batch_id': batch_id,
                'processed_at': datetime.utcnow().isoformat(),
                'responses': [asdict(response) for response in responses]
            }
            
            key = f"batch_results/{batch_id}.json"
            self.s3.put_object(
                Bucket=self.results_bucket,
                Key=key,
                Body=json.dumps(results_data, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Stored results for batch {batch_id} in S3")
            
        except Exception as e:
            logger.error(f"Error storing results for batch {batch_id}: {e}")
    
    async def _send_callbacks(self, responses: List[BatchResponse]):
        """Send callback notifications for completed requests"""
        async with aiohttp.ClientSession() as session:
            for response in responses:
                # Find original request to get callback URL
                for batch_info in self.active_batches.values():
                    for req in batch_info['requests']:
                        if req.request_id == response.request_id and req.callback_url:
                            try:
                                await session.post(
                                    req.callback_url,
                                    json=asdict(response),
                                    timeout=aiohttp.ClientTimeout(total=30)
                                )
                            except Exception as e:
                                logger.warning(f"Failed to send callback for {req.request_id}: {e}")
    
    async def _process_batches(self):
        """Periodically process accumulated batches"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check for batches that should be processed due to timeout
                expired_batches = []
                for batch_key, batch_info in self.batch_accumulator.items():
                    if current_time - batch_info['created_at'] > 30:  # 30 second timeout
                        expired_batches.append(batch_key)
                
                for batch_key in expired_batches:
                    await self._submit_batch(batch_key)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_expired_batches(self):
        """Clean up expired active batches"""
        while self.running:
            try:
                current_time = time.time()
                expired_batches = []
                
                for batch_id, batch_info in self.active_batches.items():
                    if current_time - batch_info['started_at'] > self.batch_timeout:
                        expired_batches.append(batch_id)
                
                for batch_id in expired_batches:
                    await self._handle_batch_error(batch_id, "Batch processing timeout")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _report_metrics(self):
        """Report worker metrics"""
        while self.running:
            try:
                uptime = time.time() - self.metrics['start_time']
                avg_latency = (
                    self.metrics['total_latency_ms'] / max(self.metrics['batches_completed'], 1)
                )
                
                logger.info(
                    f"Worker metrics - "
                    f"Uptime: {uptime:.1f}s, "
                    f"Requests: {self.metrics['requests_processed']}, "
                    f"Batches: {self.metrics['batches_completed']}, "
                    f"Errors: {self.metrics['errors']}, "
                    f"Avg Latency: {avg_latency:.1f}ms, "
                    f"Active Batches: {len(self.active_batches)}, "
                    f"Pending Batches: {len(self.batch_accumulator)}"
                )
                
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                logger.error(f"Error reporting metrics: {e}")
                await asyncio.sleep(60)
    
    async def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down SQS worker...")
        
        # Wait for active batches to complete (with timeout)
        shutdown_timeout = 300  # 5 minutes
        start_time = time.time()
        
        while self.active_batches and (time.time() - start_time) < shutdown_timeout:
            logger.info(f"Waiting for {len(self.active_batches)} active batches to complete...")
            await asyncio.sleep(5)
        
        # Force shutdown remaining batches
        for batch_id in list(self.active_batches.keys()):
            await self._handle_batch_error(batch_id, "Worker shutdown")
        
        self.executor.shutdown(wait=True)
        logger.info("SQS worker shutdown complete")

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    return {
        'sqs_queue_url': os.environ.get('SQS_QUEUE_URL', 'https://sqs.us-west-2.amazonaws.com/123456789/inference-batch'),
        'results_bucket': os.environ.get('RESULTS_BUCKET', 'inference-batch-results'),
        'router_endpoint': os.environ.get('ROUTER_ENDPOINT', 'http://router:8080'),
        'aws_region': os.environ.get('AWS_REGION', 'us-west-2'),
        'max_workers': int(os.environ.get('MAX_WORKERS', '10')),
        'batch_timeout': int(os.environ.get('BATCH_TIMEOUT', '300')),
        'max_batch_size': int(os.environ.get('MAX_BATCH_SIZE', '32')),
        'poll_interval': int(os.environ.get('POLL_INTERVAL', '5'))
    }

async def main():
    """Main entry point"""
    config = load_config()
    worker = SQSWorker(config)
    
    logger.info("Starting SQS batch worker...")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
