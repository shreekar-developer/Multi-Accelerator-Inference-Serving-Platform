#!/usr/bin/env python3
"""
Load Testing Script
Stress tests the inference serving platform with configurable load patterns
"""

import argparse
import asyncio
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import aiohttp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    target_url: str
    concurrent_users: int = 10
    duration_seconds: int = 300
    ramp_up_seconds: int = 60
    ramp_down_seconds: int = 60
    request_rate_qps: Optional[float] = None
    models: List[str] = None
    sla_tiers: List[str] = None
    input_data_patterns: List[str] = None
    think_time_ms: Tuple[int, int] = (100, 1000)
    timeout_seconds: int = 30
    
    def __post_init__(self):
        if self.models is None:
            self.models = ["resnet50", "distilbert", "simple_cnn"]
        if self.sla_tiers is None:
            self.sla_tiers = ["gold", "silver", "bronze"]
        if self.input_data_patterns is None:
            self.input_data_patterns = ["vision", "nlp", "small"]

@dataclass
class LoadTestResult:
    """Results from load testing"""
    timestamp: float
    user_id: int
    request_id: str
    model_name: str
    sla_tier: str
    
    # Request timing
    start_time: float
    end_time: float
    response_time_ms: float
    think_time_ms: float
    
    # Response details
    status_code: int
    success: bool
    error_message: Optional[str] = None
    
    # Response content
    accelerator_used: Optional[str] = None
    estimated_latency_ms: Optional[float] = None
    cost_per_request: Optional[float] = None
    queue_position: Optional[int] = None
    
    # Validation
    sla_met: Optional[bool] = None
    expected_latency_bound: Optional[float] = None

class LoadTester:
    """Handles load testing of the inference serving platform"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[LoadTestResult] = []
        self.active_users = 0
        self.total_requests = 0
        self.start_time = 0.0
        self.stop_requested = False
        
        # SLA bounds for validation
        self.sla_bounds = {
            "gold": 50.0,
            "silver": 150.0,
            "bronze": float('inf')
        }
    
    async def run_load_test(self) -> List[LoadTestResult]:
        """Run complete load test"""
        logger.info(f"Starting load test with {self.config.concurrent_users} users for {self.config.duration_seconds}s")
        
        self.start_time = time.time()
        self.results = []
        self.stop_requested = False
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=self.config.concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Start user tasks
            tasks = []
            
            # Ramp up users gradually
            for user_id in range(self.config.concurrent_users):
                delay = (user_id / self.config.concurrent_users) * self.config.ramp_up_seconds
                task = asyncio.create_task(self._user_session(session, user_id, delay))
                tasks.append(task)
            
            # Wait for test duration
            await asyncio.sleep(self.config.duration_seconds + self.config.ramp_up_seconds)
            
            # Signal stop and ramp down
            self.stop_requested = True
            logger.info("Stopping load test, allowing graceful ramp down...")
            
            # Wait for ramp down
            await asyncio.sleep(self.config.ramp_down_seconds)
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Load test completed. Generated {len(self.results)} requests")
        return self.results
    
    async def _user_session(self, session: aiohttp.ClientSession, user_id: int, initial_delay: float):
        """Simulate a single user session"""
        # Wait for ramp up delay
        await asyncio.sleep(initial_delay)
        
        self.active_users += 1
        logger.debug(f"User {user_id} started (active users: {self.active_users})")
        
        try:
            while not self.stop_requested:
                # Generate request
                request_data = self._generate_request_data()
                result = await self._make_request(session, user_id, request_data)
                
                if result:
                    self.results.append(result)
                    self.total_requests += 1
                
                # Think time between requests
                think_time = random.uniform(*self.config.think_time_ms) / 1000
                await asyncio.sleep(think_time)
                
                # Rate limiting if configured
                if self.config.request_rate_qps:
                    rate_delay = 1.0 / self.config.request_rate_qps
                    await asyncio.sleep(rate_delay)
        
        except asyncio.CancelledError:
            logger.debug(f"User {user_id} cancelled")
        
        except Exception as e:
            logger.error(f"User {user_id} error: {str(e)}")
        
        finally:
            self.active_users -= 1
            logger.debug(f"User {user_id} stopped (active users: {self.active_users})")
    
    async def _make_request(
        self, 
        session: aiohttp.ClientSession, 
        user_id: int, 
        request_data: Dict[str, Any]
    ) -> Optional[LoadTestResult]:
        """Make a single request to the inference API"""
        
        request_id = f"load_test_{user_id}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model_id": request_data["model_name"],
                "version": "v1.0",
                "sla_tier": request_data["sla_tier"],
                "inputs": request_data["inputs"],
                "request_id": request_id
            }
            
            # Make request
            async with session.post(
                f"{self.config.target_url}/api/v1/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                # Parse response
                response_data = {}
                if response.status == 200:
                    try:
                        response_data = await response.json()
                    except Exception as e:
                        logger.debug(f"Failed to parse response JSON: {str(e)}")
                
                # Validate SLA
                expected_bound = self.sla_bounds.get(request_data["sla_tier"], float('inf'))
                sla_met = response_time_ms <= expected_bound
                
                result = LoadTestResult(
                    timestamp=start_time,
                    user_id=user_id,
                    request_id=request_id,
                    model_name=request_data["model_name"],
                    sla_tier=request_data["sla_tier"],
                    start_time=start_time,
                    end_time=end_time,
                    response_time_ms=response_time_ms,
                    think_time_ms=request_data.get("think_time_ms", 0),
                    status_code=response.status,
                    success=response.status == 200,
                    error_message=None if response.status == 200 else f"HTTP {response.status}",
                    accelerator_used=response_data.get("accelerator"),
                    estimated_latency_ms=response_data.get("estimated_latency_ms"),
                    cost_per_request=response_data.get("cost_per_request"),
                    queue_position=response_data.get("queue_position"),
                    sla_met=sla_met,
                    expected_latency_bound=expected_bound
                )
                
                return result
        
        except asyncio.TimeoutError:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return LoadTestResult(
                timestamp=start_time,
                user_id=user_id,
                request_id=request_id,
                model_name=request_data["model_name"],
                sla_tier=request_data["sla_tier"],
                start_time=start_time,
                end_time=end_time,
                response_time_ms=response_time_ms,
                think_time_ms=request_data.get("think_time_ms", 0),
                status_code=408,
                success=False,
                error_message="Request timeout",
                sla_met=False,
                expected_latency_bound=self.sla_bounds.get(request_data["sla_tier"], float('inf'))
            )
        
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return LoadTestResult(
                timestamp=start_time,
                user_id=user_id,
                request_id=request_id,
                model_name=request_data["model_name"],
                sla_tier=request_data["sla_tier"],
                start_time=start_time,
                end_time=end_time,
                response_time_ms=response_time_ms,
                think_time_ms=request_data.get("think_time_ms", 0),
                status_code=0,
                success=False,
                error_message=str(e),
                sla_met=False,
                expected_latency_bound=self.sla_bounds.get(request_data["sla_tier"], float('inf'))
            )
    
    def _generate_request_data(self) -> Dict[str, Any]:
        """Generate realistic request data"""
        model_name = random.choice(self.config.models)
        sla_tier = random.choice(self.config.sla_tiers)
        input_pattern = random.choice(self.config.input_data_patterns)
        
        # Generate inputs based on pattern
        if input_pattern == "vision":
            inputs = [{"image_data": np.random.randn(224, 224, 3).tolist()}]
        elif input_pattern == "nlp":
            texts = [
                "This is a sample text for sentiment analysis",
                "The weather is beautiful today",
                "I love this product, it's amazing!",
                "The service was disappointing",
                "Machine learning is transforming the world"
            ]
            inputs = [{"text": random.choice(texts)}]
        else:  # small
            inputs = [{"data": np.random.randn(10).tolist()}]
        
        return {
            "model_name": model_name,
            "sla_tier": sla_tier,
            "input_pattern": input_pattern,
            "inputs": inputs
        }

class LoadTestAnalyzer:
    """Analyzes load test results and generates reports"""
    
    def __init__(self, results: List[LoadTestResult]):
        self.results = results
        self.df = pd.DataFrame([asdict(result) for result in results])
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        successful_requests = self.df[self.df['success'] == True]
        
        summary = {
            'total_requests': len(self.df),
            'successful_requests': len(successful_requests),
            'failed_requests': len(self.df) - len(successful_requests),
            'success_rate_percent': len(successful_requests) / len(self.df) * 100 if len(self.df) > 0 else 0,
            
            'response_time_stats': {
                'mean_ms': successful_requests['response_time_ms'].mean() if len(successful_requests) > 0 else 0,
                'p50_ms': successful_requests['response_time_ms'].quantile(0.5) if len(successful_requests) > 0 else 0,
                'p95_ms': successful_requests['response_time_ms'].quantile(0.95) if len(successful_requests) > 0 else 0,
                'p99_ms': successful_requests['response_time_ms'].quantile(0.99) if len(successful_requests) > 0 else 0,
                'min_ms': successful_requests['response_time_ms'].min() if len(successful_requests) > 0 else 0,
                'max_ms': successful_requests['response_time_ms'].max() if len(successful_requests) > 0 else 0,
            },
            
            'throughput': {
                'requests_per_second': len(successful_requests) / (self.df['timestamp'].max() - self.df['timestamp'].min()) if len(self.df) > 1 else 0,
                'total_duration_seconds': self.df['timestamp'].max() - self.df['timestamp'].min() if len(self.df) > 1 else 0
            },
            
            'sla_compliance': {},
            'accelerator_usage': {},
            'error_analysis': {}
        }
        
        # SLA compliance by tier
        for sla_tier in ['gold', 'silver', 'bronze']:
            tier_requests = successful_requests[successful_requests['sla_tier'] == sla_tier]
            if len(tier_requests) > 0:
                sla_met_count = tier_requests['sla_met'].sum()
                summary['sla_compliance'][sla_tier] = {
                    'total_requests': len(tier_requests),
                    'sla_met': int(sla_met_count),
                    'sla_compliance_percent': (sla_met_count / len(tier_requests)) * 100
                }
        
        # Accelerator usage
        if 'accelerator_used' in successful_requests.columns:
            accelerator_counts = successful_requests['accelerator_used'].value_counts()
            summary['accelerator_usage'] = accelerator_counts.to_dict()
        
        # Error analysis
        error_requests = self.df[self.df['success'] == False]
        if len(error_requests) > 0:
            error_counts = error_requests['error_message'].value_counts()
            summary['error_analysis'] = error_counts.to_dict()
        
        return summary
    
    def generate_plots(self, output_dir: str):
        """Generate visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Response time over time
        plt.figure(figsize=(12, 6))
        successful_df = self.df[self.df['success'] == True]
        if len(successful_df) > 0:
            plt.scatter(successful_df['timestamp'] - successful_df['timestamp'].min(), 
                       successful_df['response_time_ms'], alpha=0.6, s=20)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Response Time (ms)')
            plt.title('Response Time Over Time')
            plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'response_time_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Response time distribution by SLA tier
        plt.figure(figsize=(12, 8))
        if len(successful_df) > 0:
            for i, sla_tier in enumerate(['gold', 'silver', 'bronze']):
                tier_data = successful_df[successful_df['sla_tier'] == sla_tier]['response_time_ms']
                if len(tier_data) > 0:
                    plt.subplot(2, 2, i + 1)
                    plt.hist(tier_data, bins=50, alpha=0.7, label=sla_tier)
                    plt.axvline(tier_data.quantile(0.95), color='red', linestyle='--', label='P95')
                    plt.xlabel('Response Time (ms)')
                    plt.ylabel('Frequency')
                    plt.title(f'{sla_tier.title()} Tier Response Times')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'response_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. SLA compliance
        plt.figure(figsize=(10, 6))
        sla_compliance = []
        for sla_tier in ['gold', 'silver', 'bronze']:
            tier_requests = successful_df[successful_df['sla_tier'] == sla_tier]
            if len(tier_requests) > 0:
                compliance_rate = tier_requests['sla_met'].mean() * 100
                sla_compliance.append({'SLA Tier': sla_tier.title(), 'Compliance %': compliance_rate})
        
        if sla_compliance:
            compliance_df = pd.DataFrame(sla_compliance)
            bars = plt.bar(compliance_df['SLA Tier'], compliance_df['Compliance %'])
            plt.ylabel('SLA Compliance (%)')
            plt.title('SLA Compliance by Tier')
            plt.ylim(0, 100)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'sla_compliance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Accelerator usage
        plt.figure(figsize=(10, 6))
        if 'accelerator_used' in successful_df.columns and successful_df['accelerator_used'].notna().any():
            accelerator_counts = successful_df['accelerator_used'].value_counts()
            if len(accelerator_counts) > 0:
                plt.pie(accelerator_counts.values, labels=accelerator_counts.index, autopct='%1.1f%%')
                plt.title('Accelerator Usage Distribution')
        plt.savefig(output_dir / 'accelerator_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Throughput over time
        plt.figure(figsize=(12, 6))
        if len(successful_df) > 0:
            # Calculate throughput in 10-second windows
            start_time = successful_df['timestamp'].min()
            end_time = successful_df['timestamp'].max()
            window_size = 10  # seconds
            
            times = []
            throughputs = []
            
            current_time = start_time
            while current_time < end_time:
                window_requests = successful_df[
                    (successful_df['timestamp'] >= current_time) & 
                    (successful_df['timestamp'] < current_time + window_size)
                ]
                throughput = len(window_requests) / window_size
                
                times.append(current_time - start_time)
                throughputs.append(throughput)
                current_time += window_size
            
            plt.plot(times, throughputs, linewidth=2)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Throughput (requests/second)')
            plt.title('Throughput Over Time (10s windows)')
            plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'throughput_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {output_dir}")

def save_results(results: List[LoadTestResult], output_file: str):
    """Save load test results to JSON file"""
    results_data = [asdict(result) for result in results]
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_file}")

def load_results(input_file: str) -> List[LoadTestResult]:
    """Load load test results from JSON file"""
    with open(input_file, 'r') as f:
        results_data = json.load(f)
    
    results = []
    for result_data in results_data:
        result = LoadTestResult(**result_data)
        results.append(result)
    
    return results

async def main():
    parser = argparse.ArgumentParser(description='Load test the inference serving platform')
    parser.add_argument('--target-url', required=True, help='Target URL for load testing')
    parser.add_argument('--concurrent-users', type=int, default=10, help='Number of concurrent users')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds')
    parser.add_argument('--ramp-up', type=int, default=60, help='Ramp up time in seconds')
    parser.add_argument('--ramp-down', type=int, default=60, help='Ramp down time in seconds')
    parser.add_argument('--request-rate', type=float, help='Target request rate (QPS)')
    parser.add_argument('--models', nargs='+', default=['resnet50', 'distilbert', 'simple_cnn'])
    parser.add_argument('--sla-tiers', nargs='+', default=['gold', 'silver', 'bronze'])
    parser.add_argument('--think-time-min', type=int, default=100, help='Min think time in ms')
    parser.add_argument('--think-time-max', type=int, default=1000, help='Max think time in ms')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    parser.add_argument('--output-dir', default='load_test_results', help='Output directory')
    parser.add_argument('--analyze-only', help='Analyze existing results file')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.analyze_only:
        # Analyze existing results
        logger.info(f"Analyzing existing results from {args.analyze_only}")
        results = load_results(args.analyze_only)
    else:
        # Run load test
        config = LoadTestConfig(
            target_url=args.target_url,
            concurrent_users=args.concurrent_users,
            duration_seconds=args.duration,
            ramp_up_seconds=args.ramp_up,
            ramp_down_seconds=args.ramp_down,
            request_rate_qps=args.request_rate,
            models=args.models,
            sla_tiers=args.sla_tiers,
            think_time_ms=(args.think_time_min, args.think_time_max),
            timeout_seconds=args.timeout
        )
        
        tester = LoadTester(config)
        results = await tester.run_load_test()
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"load_test_results_{timestamp}.json"
        save_results(results, str(results_file))
    
    # Analyze results
    analyzer = LoadTestAnalyzer(results)
    summary = analyzer.generate_summary()
    
    # Save summary
    summary_file = output_dir / f"load_test_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Generate plots
    analyzer.generate_plots(str(output_dir))
    
    # Print summary
    print("\n=== Load Test Summary ===")
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Successful: {summary['successful_requests']}")
    print(f"Failed: {summary['failed_requests']}")
    print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
    print(f"Average Response Time: {summary['response_time_stats']['mean_ms']:.1f} ms")
    print(f"P95 Response Time: {summary['response_time_stats']['p95_ms']:.1f} ms")
    print(f"P99 Response Time: {summary['response_time_stats']['p99_ms']:.1f} ms")
    print(f"Throughput: {summary['throughput']['requests_per_second']:.1f} req/s")
    
    print("\n=== SLA Compliance ===")
    for tier, compliance in summary['sla_compliance'].items():
        print(f"{tier.title()}: {compliance['sla_compliance_percent']:.1f}% "
              f"({compliance['sla_met']}/{compliance['total_requests']})")
    
    if summary['accelerator_usage']:
        print("\n=== Accelerator Usage ===")
        for accelerator, count in summary['accelerator_usage'].items():
            percentage = (count / summary['successful_requests']) * 100
            print(f"{accelerator}: {count} requests ({percentage:.1f}%)")
    
    if summary['error_analysis']:
        print("\n=== Error Analysis ===")
        for error, count in summary['error_analysis'].items():
            print(f"{error}: {count} occurrences")
    
    print(f"\nResults and plots saved to: {output_dir}")

if __name__ == '__main__':
    asyncio.run(main()) 