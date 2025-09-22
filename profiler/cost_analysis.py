#!/usr/bin/env python3
"""
Cost Analysis and Optimization Tool
Analyzes cost efficiency across accelerators and provides optimization recommendations.
"""

import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AcceleratorCost:
    """Cost configuration for different accelerator types"""
    accelerator: str
    instance_type: str
    cost_per_hour: float
    cores_or_units: int
    memory_gb: int

@dataclass
class ModelCostProfile:
    """Cost profile for a specific model on an accelerator"""
    model_id: str
    version: str
    accelerator: str
    batch_size: int
    qps_sustained: float
    p95_latency_ms: float
    cost_per_hour: float
    cost_per_1k_requests: float
    efficiency_score: float

class CostAnalyzer:
    """Analyzes cost efficiency and provides optimization recommendations"""
    
    def __init__(self, region: str = "us-west-2"):
        self.region = region
        self.dynamodb = boto3.client('dynamodb', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        
        # AWS instance costs (on-demand pricing as of 2024)
        self.accelerator_costs = {
            "cpu": AcceleratorCost("cpu", "c7g.4xlarge", 0.6877, 16, 32),
            "gpu": AcceleratorCost("gpu", "g5.2xlarge", 1.212, 1, 32),
            "inferentia": AcceleratorCost("inferentia", "inf2.xlarge", 0.7584, 1, 16)
        }
    
    def get_model_profiles(self, model_id: str = None) -> List[Dict]:
        """Retrieve model profiles from DynamoDB"""
        try:
            table_name = "model-profiles"
            
            if model_id:
                response = self.dynamodb.query(
                    TableName=table_name,
                    KeyConditionExpression="PK = :pk",
                    ExpressionAttributeValues={":pk": {"S": model_id}}
                )
            else:
                response = self.dynamodb.scan(TableName=table_name)
            
            profiles = []
            for item in response.get('Items', []):
                profile = {
                    'model_id': item['PK']['S'].split('#')[0],
                    'version': item['PK']['S'].split('#')[1] if '#' in item['PK']['S'] else 'latest',
                    'accelerator': item['SK']['S'].split('#')[0],
                    'batch_size': int(item['SK']['S'].split('#')[1]),
                    'seq_len': int(item['SK']['S'].split('#')[2]) if len(item['SK']['S'].split('#')) > 2 else 128,
                    'p50_ms': float(item.get('p50_ms', {}).get('N', 0)),
                    'p95_ms': float(item.get('p95_ms', {}).get('N', 0)),
                    'qps_sustained': float(item.get('qps_sustained', {}).get('N', 0)),
                    'mem_mb': int(item.get('mem_mb', {}).get('N', 0)),
                    'last_updated': item.get('last_updated', {}).get('S', '')
                }
                profiles.append(profile)
            
            return profiles
            
        except Exception as e:
            logger.error(f"Error retrieving model profiles: {e}")
            return []
    
    def calculate_cost_per_request(self, profile: Dict) -> float:
        """Calculate cost per 1000 requests for a given profile"""
        accelerator = profile['accelerator']
        qps = profile['qps_sustained']
        
        if qps <= 0:
            return float('inf')
        
        cost_config = self.accelerator_costs.get(accelerator)
        if not cost_config:
            return float('inf')
        
        # Cost per request = (cost_per_hour / qps) / 3600 * 1000
        cost_per_1k_requests = (cost_config.cost_per_hour / qps) * (1000 / 3600)
        return cost_per_1k_requests
    
    def calculate_efficiency_score(self, profile: Dict) -> float:
        """Calculate efficiency score (higher is better)"""
        cost_per_1k = self.calculate_cost_per_request(profile)
        latency_ms = profile['p95_ms']
        
        if cost_per_1k == float('inf') or latency_ms <= 0:
            return 0.0
        
        # Efficiency = throughput / (cost * latency_penalty)
        # Lower cost and latency = higher efficiency
        latency_penalty = max(1.0, latency_ms / 100.0)  # Normalize to 100ms baseline
        efficiency = (profile['qps_sustained'] * 1000) / (cost_per_1k * latency_penalty)
        return efficiency
    
    def analyze_model_costs(self, model_id: str = None) -> List[ModelCostProfile]:
        """Analyze cost profiles for models"""
        profiles = self.get_model_profiles(model_id)
        cost_profiles = []
        
        for profile in profiles:
            cost_per_1k = self.calculate_cost_per_request(profile)
            efficiency = self.calculate_efficiency_score(profile)
            
            cost_profile = ModelCostProfile(
                model_id=profile['model_id'],
                version=profile['version'],
                accelerator=profile['accelerator'],
                batch_size=profile['batch_size'],
                qps_sustained=profile['qps_sustained'],
                p95_latency_ms=profile['p95_ms'],
                cost_per_hour=self.accelerator_costs[profile['accelerator']].cost_per_hour,
                cost_per_1k_requests=cost_per_1k,
                efficiency_score=efficiency
            )
            cost_profiles.append(cost_profile)
        
        return cost_profiles
    
    def get_live_metrics(self, hours_back: int = 24) -> Dict:
        """Get live metrics from CloudWatch"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        try:
            # Get request volume by accelerator
            response = self.cloudwatch.get_metric_statistics(
                Namespace='InferenceRouter',
                MetricName='RequestCount',
                Dimensions=[{'Name': 'Accelerator', 'Value': 'all'}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour periods
                Statistics=['Sum']
            )
            
            total_requests = sum([point['Sum'] for point in response['Datapoints']])
            
            # Get cost metrics by accelerator
            cost_metrics = {}
            for accelerator in ['cpu', 'gpu', 'inferentia']:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='InferenceRouter',
                    MetricName='ActiveInstances',
                    Dimensions=[{'Name': 'Accelerator', 'Value': accelerator}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,
                    Statistics=['Average']
                )
                
                avg_instances = np.mean([point['Average'] for point in response['Datapoints']]) if response['Datapoints'] else 0
                cost_config = self.accelerator_costs[accelerator]
                hourly_cost = avg_instances * cost_config.cost_per_hour
                
                cost_metrics[accelerator] = {
                    'avg_instances': avg_instances,
                    'hourly_cost': hourly_cost,
                    'total_cost_24h': hourly_cost * hours_back
                }
            
            return {
                'total_requests': total_requests,
                'cost_by_accelerator': cost_metrics,
                'period_hours': hours_back
            }
            
        except Exception as e:
            logger.warning(f"Could not retrieve live metrics: {e}")
            return {'total_requests': 0, 'cost_by_accelerator': {}, 'period_hours': hours_back}
    
    def generate_optimization_recommendations(self, cost_profiles: List[ModelCostProfile]) -> List[Dict]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Group by model
        model_groups = {}
        for profile in cost_profiles:
            key = f"{profile.model_id}#{profile.version}"
            if key not in model_groups:
                model_groups[key] = []
            model_groups[key].append(profile)
        
        for model_key, profiles in model_groups.items():
            model_id, version = model_key.split('#')
            
            # Sort by efficiency score (highest first)
            profiles.sort(key=lambda x: x.efficiency_score, reverse=True)
            
            if len(profiles) > 1:
                best = profiles[0]
                worst = profiles[-1]
                
                if best.efficiency_score > worst.efficiency_score * 1.2:  # 20% better
                    savings = worst.cost_per_1k_requests - best.cost_per_1k_requests
                    savings_pct = (savings / worst.cost_per_1k_requests) * 100
                    
                    recommendations.append({
                        'type': 'accelerator_switch',
                        'model_id': model_id,
                        'version': version,
                        'current_accelerator': worst.accelerator,
                        'recommended_accelerator': best.accelerator,
                        'cost_savings_per_1k': savings,
                        'savings_percentage': savings_pct,
                        'latency_impact_ms': best.p95_latency_ms - worst.p95_latency_ms,
                        'priority': 'high' if savings_pct > 30 else 'medium'
                    })
            
            # Check for batch size optimization
            accelerator_groups = {}
            for profile in profiles:
                if profile.accelerator not in accelerator_groups:
                    accelerator_groups[profile.accelerator] = []
                accelerator_groups[profile.accelerator].append(profile)
            
            for accelerator, accel_profiles in accelerator_groups.items():
                if len(accel_profiles) > 1:
                    accel_profiles.sort(key=lambda x: x.efficiency_score, reverse=True)
                    best_batch = accel_profiles[0]
                    
                    for profile in accel_profiles[1:]:
                        if best_batch.efficiency_score > profile.efficiency_score * 1.15:
                            recommendations.append({
                                'type': 'batch_size_optimization',
                                'model_id': model_id,
                                'version': version,
                                'accelerator': accelerator,
                                'current_batch_size': profile.batch_size,
                                'recommended_batch_size': best_batch.batch_size,
                                'efficiency_improvement': best_batch.efficiency_score - profile.efficiency_score,
                                'priority': 'medium'
                            })
        
        return recommendations
    
    def create_cost_report(self, output_file: str = None) -> Dict:
        """Create comprehensive cost analysis report"""
        logger.info("Generating cost analysis report...")
        
        # Analyze all models
        cost_profiles = self.analyze_model_costs()
        
        # Get live metrics
        live_metrics = self.get_live_metrics()
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations(cost_profiles)
        
        # Calculate summary statistics
        total_models = len(set(f"{p.model_id}#{p.version}" for p in cost_profiles))
        avg_cost_per_1k = np.mean([p.cost_per_1k_requests for p in cost_profiles if p.cost_per_1k_requests != float('inf')])
        
        cost_by_accelerator = {}
        for accelerator in ['cpu', 'gpu', 'inferentia']:
            accel_profiles = [p for p in cost_profiles if p.accelerator == accelerator]
            if accel_profiles:
                cost_by_accelerator[accelerator] = {
                    'avg_cost_per_1k': np.mean([p.cost_per_1k_requests for p in accel_profiles if p.cost_per_1k_requests != float('inf')]),
                    'avg_efficiency': np.mean([p.efficiency_score for p in accel_profiles]),
                    'model_count': len(accel_profiles)
                }
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_models': total_models,
                'total_configurations': len(cost_profiles),
                'avg_cost_per_1k_requests': avg_cost_per_1k,
                'cost_by_accelerator': cost_by_accelerator
            },
            'live_metrics': live_metrics,
            'model_profiles': [
                {
                    'model_id': p.model_id,
                    'version': p.version,
                    'accelerator': p.accelerator,
                    'batch_size': p.batch_size,
                    'cost_per_1k_requests': p.cost_per_1k_requests,
                    'efficiency_score': p.efficiency_score,
                    'p95_latency_ms': p.p95_latency_ms,
                    'qps_sustained': p.qps_sustained
                } for p in cost_profiles
            ],
            'recommendations': recommendations,
            'potential_savings': sum([r.get('cost_savings_per_1k', 0) for r in recommendations if r['type'] == 'accelerator_switch'])
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Cost report saved to {output_file}")
        
        return report
    
    def create_cost_visualization(self, cost_profiles: List[ModelCostProfile], output_dir: str = "plots"):
        """Create cost visualization plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([
            {
                'model_id': p.model_id,
                'accelerator': p.accelerator,
                'batch_size': p.batch_size,
                'cost_per_1k': p.cost_per_1k_requests if p.cost_per_1k_requests != float('inf') else None,
                'efficiency': p.efficiency_score,
                'latency_ms': p.p95_latency_ms,
                'qps': p.qps_sustained
            } for p in cost_profiles
        ]).dropna()
        
        # Cost comparison by accelerator
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='accelerator', y='cost_per_1k')
        plt.title('Cost per 1K Requests by Accelerator Type')
        plt.ylabel('Cost per 1K Requests ($)')
        plt.xlabel('Accelerator Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cost_by_accelerator.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Efficiency vs Cost scatter
        plt.figure(figsize=(10, 8))
        for accelerator in df['accelerator'].unique():
            accel_data = df[df['accelerator'] == accelerator]
            plt.scatter(accel_data['cost_per_1k'], accel_data['efficiency'], 
                       label=accelerator, alpha=0.7, s=60)
        
        plt.xlabel('Cost per 1K Requests ($)')
        plt.ylabel('Efficiency Score')
        plt.title('Cost vs Efficiency by Accelerator')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cost_vs_efficiency.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Latency vs Cost
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['cost_per_1k'], df['latency_ms'], 
                            c=df['accelerator'].astype('category').cat.codes, 
                            s=df['qps']*2, alpha=0.6, cmap='viridis')
        plt.xlabel('Cost per 1K Requests ($)')
        plt.ylabel('P95 Latency (ms)')
        plt.title('Cost vs Latency (bubble size = QPS)')
        plt.colorbar(scatter, label='Accelerator Type')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cost_vs_latency.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cost visualization plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Cost Analysis and Optimization Tool")
    parser.add_argument("--model-id", help="Specific model ID to analyze")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--output", default="cost_report.json", help="Output report file")
    parser.add_argument("--plots", action="store_true", help="Generate visualization plots")
    parser.add_argument("--plot-dir", default="plots", help="Directory for plots")
    
    args = parser.parse_args()
    
    analyzer = CostAnalyzer(region=args.region)
    
    # Generate cost report
    report = analyzer.create_cost_report(args.output)
    
    print("\n=== COST ANALYSIS SUMMARY ===")
    print(f"Total Models: {report['summary']['total_models']}")
    print(f"Total Configurations: {report['summary']['total_configurations']}")
    print(f"Average Cost per 1K Requests: ${report['summary']['avg_cost_per_1k_requests']:.4f}")
    
    print("\n=== COST BY ACCELERATOR ===")
    for accelerator, stats in report['summary']['cost_by_accelerator'].items():
        print(f"{accelerator.upper()}:")
        print(f"  Average Cost per 1K: ${stats['avg_cost_per_1k']:.4f}")
        print(f"  Average Efficiency: {stats['avg_efficiency']:.2f}")
        print(f"  Model Count: {stats['model_count']}")
    
    print(f"\n=== OPTIMIZATION RECOMMENDATIONS ({len(report['recommendations'])}) ===")
    for i, rec in enumerate(report['recommendations'][:5], 1):  # Show top 5
        print(f"{i}. {rec['type'].replace('_', ' ').title()}")
        if rec['type'] == 'accelerator_switch':
            print(f"   Model: {rec['model_id']}")
            print(f"   Switch from {rec['current_accelerator']} to {rec['recommended_accelerator']}")
            print(f"   Savings: ${rec['cost_savings_per_1k']:.4f} per 1K requests ({rec['savings_percentage']:.1f}%)")
        elif rec['type'] == 'batch_size_optimization':
            print(f"   Model: {rec['model_id']} on {rec['accelerator']}")
            print(f"   Change batch size from {rec['current_batch_size']} to {rec['recommended_batch_size']}")
    
    if len(report['recommendations']) > 5:
        print(f"   ... and {len(report['recommendations']) - 5} more recommendations")
    
    print(f"\nTotal Potential Savings: ${report['potential_savings']:.4f} per 1K requests")
    
    # Generate plots if requested
    if args.plots:
        cost_profiles = analyzer.analyze_model_costs(args.model_id)
        analyzer.create_cost_visualization(cost_profiles, args.plot_dir)
    
    print(f"\nDetailed report saved to: {args.output}")

if __name__ == "__main__":
    main()
