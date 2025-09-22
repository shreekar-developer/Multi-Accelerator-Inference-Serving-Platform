#!/usr/bin/env python3
"""
Kafka Async Worker for High-Throughput Batch Inference Processing
Processes batch inference requests from Kafka topics with advanced batching and partitioning.
"""

import json
import asyncio
import aiohttp
import logging
import os
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class KafkaBatchRequest:
    """Kafka batch inference request structure"""
    request_id: str
    model_id: str
    version: str
    sla_tier: str
    inputs: List[Dict[str, Any]]
    partition_key: Optional[str] = None
    priority: int = 0
    submitted_at: str = ""
    max_batch_size: int = 64
    timeout_seconds: int = 300
    correlation_id: Optional[str] = None

@dataclass
class KafkaBatchResponse:
    """Kafka batch inference response structure"""
    request_id: str
    correlation_id: Optional[str]
    status: str  # "completed", "failed", "partial"
    outputs: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processed_at: str
    latency_ms: float
    accelerator_used: str
    cost_estimate: float
    partition_key: Optional[str] = None

class KafkaWorker:
    """Kafka Worker for processing high-throughput batch inference requests"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bootstrap_servers = config['kafka_bootstrap_servers']
        self.input_topic = config['input_topic']
        self.output_topic = config['output_topic']
        self.consumer_group = config['consumer_group']
        self.router_endpoint = config.get('router_endpoint', 'http://router:8080')
        
        # Batching configuration
        self.max_batch_size = config.get('max_batch_size', 64)
        self.batch_timeout_ms = config.get('batch_timeout_ms', 30000)
        self.max_workers = config.get('max_workers', 20)
        self.prefetch_count = config.get('prefetch_count', 100)
        
        # Kafka configuration
        self.kafka_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': False,
            'group_id': self.consumer_group,
            'max_poll_records': self.prefetch_count,
            'fetch_max_wait_ms': 1000,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'key_deserializer': lambda m: m.decode('utf-8') if m else None
        }
        
        self.producer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None,
            'acks': 'all',
            'retries': 3,
            'batch_size': 16384,
            'linger_ms': 10,
            'compression_type': 'gzip'
        }
        
        self.consumer = None
        self.producer = None
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Batch management
        self.batch_accumulator = {}  # partition -> batch_info
        self.active_batches = {}
        self.partition_assignments = set()
        
        # Metrics
        self.metrics = {
            'messages_consumed': 0,
            'messages_produced': 0,
            'batches_processed': 0,
            'errors': 0,
            'total_latency_ms': 0,
            'start_time': time.time(),
            'last_commit_time': time.time()
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def start(self):
        """Start the Kafka worker"""
        logger.info("Starting Kafka worker...")
        
        try:
            # Initialize Kafka consumer and producer
            await self._initialize_kafka()
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._process_batches()),
                asyncio.create_task(self._commit_offsets()),
                asyncio.create_task(self._report_metrics())
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            await self._shutdown()
    
    async def _initialize_kafka(self):
        """Initialize Kafka consumer and producer"""
        try:
            # Create consumer
            self.consumer = KafkaConsumer(
                self.input_topic,
                **self.kafka_config
            )
            
            # Create producer
            self.producer = KafkaProducer(**self.producer_config)
            
            # Get partition assignments
            partitions = self.consumer.partitions_for_topic(self.input_topic)
            if partitions:
                self.partition_assignments = partitions
                logger.info(f"Assigned to partitions: {sorted(partitions)}")
            
            logger.info("Kafka client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            raise
    
    async def _consume_messages(self):
        """Consume messages from Kafka topic"""
        while self.running:
            try:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000, max_records=self.prefetch_count)
                
                if message_batch:
                    await self._process_message_batch(message_batch)
                
            except KafkaError as e:
                logger.error(f"Kafka error: {e}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error consuming messages: {e}")
                await asyncio.sleep(1)
    
    async def _process_message_batch(self, message_batch: Dict):
        """Process a batch of messages from Kafka"""
        for topic_partition, messages in message_batch.items():
            partition = topic_partition.partition
            
            for message in messages:
                try:
                    # Parse message
                    request_data = message.value
                    batch_request = KafkaBatchRequest(**request_data)
                    
                    # Add to partition-specific batch
                    await self._add_to_partition_batch(partition, batch_request, message)
                    
                    self.metrics['messages_consumed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self.metrics['errors'] += 1
    
    async def _add_to_partition_batch(self, partition: int, request: KafkaBatchRequest, message):
        """Add request to partition-specific batch for optimal processing"""
        # Use model+version+sla as sub-batch key within partition
        batch_key = f"p{partition}_{request.model_id}#{request.version}#{request.sla_tier}"
        
        if batch_key not in self.batch_accumulator:
            self.batch_accumulator[batch_key] = {
                'requests': [],
                'messages': [],
                'created_at': time.time(),
                'partition': partition,
                'model_id': request.model_id,
                'version': request.version,
                'sla_tier': request.sla_tier,
                'last_offset': message.offset
            }
        
        batch_info = self.batch_accumulator[batch_key]
        batch_info['requests'].append(request)
        batch_info['messages'].append(message)
        batch_info['last_offset'] = max(batch_info['last_offset'], message.offset)
        
        # Check if batch should be processed
        should_process = (
            len(batch_info['requests']) >= self.max_batch_size or
            (time.time() - batch_info['created_at']) * 1000 > self.batch_timeout_ms or
            request.sla_tier == 'gold' or  # Process gold tier quickly
            request.priority > 5  # High priority requests
        )
        
        if should_process:
            await self._submit_partition_batch(batch_key)
    
    async def _submit_partition_batch(self, batch_key: str):
        """Submit partition batch for processing"""
        if batch_key not in self.batch_accumulator:
            return
        
        batch_info = self.batch_accumulator.pop(batch_key)
        requests = batch_info['requests']
        messages = batch_info['messages']
        
        if not requests:
            return
        
        batch_id = str(uuid.uuid4())
        logger.info(f"Submitting partition batch {batch_id} with {len(requests)} requests from partition {batch_info['partition']}")
        
        # Create batch processing task
        self.active_batches[batch_id] = {
            'requests': requests,
            'messages': messages,
            'started_at': time.time(),
            'partition': batch_info['partition'],
            'model_id': batch_info['model_id'],
            'version': batch_info['version'],
            'sla_tier': batch_info['sla_tier'],
            'last_offset': batch_info['last_offset']
        }
        
        # Process batch asynchronously
        asyncio.create_task(self._process_partition_batch(batch_id))
    
    async def _process_partition_batch(self, batch_id: str):
        """Process a partition batch of inference requests"""
        if batch_id not in self.active_batches:
            return
        
        batch_info = self.active_batches[batch_id]
        requests = batch_info['requests']
        start_time = time.time()
        
        try:
            # Smart batching: group similar inputs together
            input_groups = self._group_inputs_by_similarity(requests)
            all_responses = []
            
            for group_inputs, group_requests in input_groups:
                # Create inference request for this group
                inference_payload = {
                    'model_id': batch_info['model_id'],
                    'version': batch_info['version'],
                    'sla_tier': 'bronze',  # Batch processing is typically bronze
                    'inputs': group_inputs,
                    'batch_mode': True,
                    'batch_id': batch_id,
                    'partition': batch_info['partition']
                }
                
                # Send to router
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.router_endpoint}/predict",
                        json=inference_payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            group_responses = await self._create_group_responses(
                                group_requests, result, batch_id
                            )
                            all_responses.extend(group_responses)
                        else:
                            error_text = await response.text()
                            error_responses = await self._create_error_responses(
                                group_requests, f"HTTP {response.status}: {error_text}", batch_id
                            )
                            all_responses.extend(error_responses)
            
            # Send responses to output topic
            await self._send_responses(all_responses, batch_info['partition'])
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics['batches_processed'] += 1
            self.metrics['total_latency_ms'] += processing_time
            self.metrics['messages_produced'] += len(all_responses)
            
            logger.info(f"Batch {batch_id} completed in {processing_time:.1f}ms with {len(all_responses)} responses")
            
        except asyncio.TimeoutError:
            await self._handle_batch_timeout(batch_id)
        except Exception as e:
            await self._handle_batch_error(batch_id, str(e))
        finally:
            # Clean up
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
    
    def _group_inputs_by_similarity(self, requests: List[KafkaBatchRequest]) -> List[tuple]:
        """Group inputs by similarity for optimal batch processing"""
        # Simple grouping by input structure hash
        groups = {}
        
        for request in requests:
            for input_data in request.inputs:
                # Create hash based on input structure
                input_hash = self._hash_input_structure(input_data)
                
                if input_hash not in groups:
                    groups[input_hash] = {'inputs': [], 'requests': []}
                
                groups[input_hash]['inputs'].append(input_data)
                groups[input_hash]['requests'].append(request)
        
        return [(group['inputs'], group['requests']) for group in groups.values()]
    
    def _hash_input_structure(self, input_data: Dict) -> str:
        """Create hash based on input data structure"""
        # Create a structure signature
        structure = {}
        for key, value in input_data.items():
            if isinstance(value, str):
                structure[key] = f"str({len(value)})"
            elif isinstance(value, (int, float)):
                structure[key] = type(value).__name__
            elif isinstance(value, list):
                structure[key] = f"list({len(value)})"
            elif isinstance(value, dict):
                structure[key] = f"dict({len(value)})"
            else:
                structure[key] = type(value).__name__
        
        return hashlib.md5(json.dumps(structure, sort_keys=True).encode()).hexdigest()[:8]
    
    async def _create_group_responses(self, requests: List[KafkaBatchRequest], 
                                    result: Dict, batch_id: str) -> List[KafkaBatchResponse]:
        """Create responses for a successful group"""
        outputs = result.get('outputs', [])
        responses = []
        
        output_idx = 0
        for request in requests:
            request_outputs = []
            for _ in request.inputs:
                if output_idx < len(outputs):
                    request_outputs.append(outputs[output_idx])
                    output_idx += 1
            
            response = KafkaBatchResponse(
                request_id=request.request_id,
                correlation_id=request.correlation_id,
                status="completed",
                outputs=request_outputs,
                metadata={
                    'batch_id': batch_id,
                    'model_id': request.model_id,
                    'version': request.version,
                    'partition': self.active_batches[batch_id]['partition']
                },
                processed_at=datetime.utcnow().isoformat(),
                latency_ms=result.get('latency_ms', 0),
                accelerator_used=result.get('accelerator', 'unknown'),
                cost_estimate=result.get('cost_estimate', 0.0),
                partition_key=request.partition_key
            )
            responses.append(response)
        
        return responses
    
    async def _create_error_responses(self, requests: List[KafkaBatchRequest], 
                                    error_message: str, batch_id: str) -> List[KafkaBatchResponse]:
        """Create error responses for a failed group"""
        responses = []
        
        for request in requests:
            response = KafkaBatchResponse(
                request_id=request.request_id,
                correlation_id=request.correlation_id,
                status="failed",
                outputs=[],
                metadata={
                    'batch_id': batch_id,
                    'error': error_message,
                    'model_id': request.model_id,
                    'version': request.version,
                    'partition': self.active_batches[batch_id]['partition']
                },
                processed_at=datetime.utcnow().isoformat(),
                latency_ms=0,
                accelerator_used='none',
                cost_estimate=0.0,
                partition_key=request.partition_key
            )
            responses.append(response)
        
        return responses
    
    async def _send_responses(self, responses: List[KafkaBatchResponse], source_partition: int):
        """Send responses to Kafka output topic"""
        try:
            for response in responses:
                # Use correlation_id or request_id as key for partitioning
                key = response.correlation_id or response.request_id
                
                # Send to output topic
                future = self.producer.send(
                    self.output_topic,
                    key=key,
                    value=asdict(response),
                    partition=source_partition % self.producer.partitions_for(self.output_topic).__len__()
                )
                
                # Don't wait for each message, batch them
                
            # Flush producer to ensure messages are sent
            self.producer.flush(timeout=10)
            
        except Exception as e:
            logger.error(f"Error sending responses: {e}")
            raise
    
    async def _handle_batch_timeout(self, batch_id: str):
        """Handle batch processing timeout"""
        if batch_id not in self.active_batches:
            return
        
        batch_info = self.active_batches[batch_id]
        error_responses = await self._create_error_responses(
            batch_info['requests'], 
            "Batch processing timeout", 
            batch_id
        )
        
        await self._send_responses(error_responses, batch_info['partition'])
        self.metrics['errors'] += len(batch_info['requests'])
        
        logger.error(f"Batch {batch_id} timed out")
    
    async def _handle_batch_error(self, batch_id: str, error_message: str):
        """Handle batch processing error"""
        if batch_id not in self.active_batches:
            return
        
        batch_info = self.active_batches[batch_id]
        error_responses = await self._create_error_responses(
            batch_info['requests'], 
            error_message, 
            batch_id
        )
        
        await self._send_responses(error_responses, batch_info['partition'])
        self.metrics['errors'] += len(batch_info['requests'])
        
        logger.error(f"Batch {batch_id} failed: {error_message}")
    
    async def _process_batches(self):
        """Periodically process accumulated batches"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check for batches that should be processed due to timeout
                expired_batches = []
                for batch_key, batch_info in self.batch_accumulator.items():
                    if (current_time - batch_info['created_at']) * 1000 > self.batch_timeout_ms:
                        expired_batches.append(batch_key)
                
                for batch_key in expired_batches:
                    await self._submit_partition_batch(batch_key)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _commit_offsets(self):
        """Periodically commit Kafka offsets"""
        while self.running:
            try:
                # Only commit if we have processed messages
                if time.time() - self.metrics['last_commit_time'] > 30:  # Commit every 30 seconds
                    self.consumer.commit()
                    self.metrics['last_commit_time'] = time.time()
                    logger.debug("Committed Kafka offsets")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error committing offsets: {e}")
                await asyncio.sleep(10)
    
    async def _report_metrics(self):
        """Report worker metrics"""
        while self.running:
            try:
                uptime = time.time() - self.metrics['start_time']
                avg_latency = (
                    self.metrics['total_latency_ms'] / max(self.metrics['batches_processed'], 1)
                )
                throughput = self.metrics['messages_consumed'] / max(uptime, 1)
                
                logger.info(
                    f"Kafka worker metrics - "
                    f"Uptime: {uptime:.1f}s, "
                    f"Consumed: {self.metrics['messages_consumed']}, "
                    f"Produced: {self.metrics['messages_produced']}, "
                    f"Batches: {self.metrics['batches_processed']}, "
                    f"Errors: {self.metrics['errors']}, "
                    f"Throughput: {throughput:.1f} msg/s, "
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
        logger.info("Shutting down Kafka worker...")
        
        # Process remaining batches
        for batch_key in list(self.batch_accumulator.keys()):
            await self._submit_partition_batch(batch_key)
        
        # Wait for active batches to complete (with timeout)
        shutdown_timeout = 300  # 5 minutes
        start_time = time.time()
        
        while self.active_batches and (time.time() - start_time) < shutdown_timeout:
            logger.info(f"Waiting for {len(self.active_batches)} active batches to complete...")
            await asyncio.sleep(5)
        
        # Force shutdown remaining batches
        for batch_id in list(self.active_batches.keys()):
            await self._handle_batch_error(batch_id, "Worker shutdown")
        
        # Close Kafka connections
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        
        self.executor.shutdown(wait=True)
        logger.info("Kafka worker shutdown complete")

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    return {
        'kafka_bootstrap_servers': os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        'input_topic': os.environ.get('INPUT_TOPIC', 'inference-requests'),
        'output_topic': os.environ.get('OUTPUT_TOPIC', 'inference-responses'),
        'consumer_group': os.environ.get('CONSUMER_GROUP', 'inference-workers'),
        'router_endpoint': os.environ.get('ROUTER_ENDPOINT', 'http://router:8080'),
        'max_workers': int(os.environ.get('MAX_WORKERS', '20')),
        'max_batch_size': int(os.environ.get('MAX_BATCH_SIZE', '64')),
        'batch_timeout_ms': int(os.environ.get('BATCH_TIMEOUT_MS', '30000')),
        'prefetch_count': int(os.environ.get('PREFETCH_COUNT', '100'))
    }

async def main():
    """Main entry point"""
    config = load_config()
    worker = KafkaWorker(config)
    
    logger.info("Starting Kafka batch worker...")
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
