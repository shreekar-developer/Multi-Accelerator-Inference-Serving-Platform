# ML Inference Platform - Grafana Dashboards

This directory contains pre-built Grafana dashboards for monitoring the ML Inference Platform.

## Dashboard Overview

### 1. Inference Overview Dashboard (`inference-overview-dashboard.json`)
**Purpose**: High-level operational metrics and system health
**Key Metrics**:
- Request rate by accelerator type
- P95 latency trends
- SLA compliance rates
- Request volume by model
- Error rates
- Queue depths
- Active instance counts
- Top models by volume
- Routing decision distribution

**Use Cases**:
- Daily operational monitoring
- Quick health checks
- Performance trending
- Capacity planning

### 2. Cost Optimization Dashboard (`cost-optimization-dashboard.json`)
**Purpose**: Cost analysis and optimization opportunities
**Key Metrics**:
- Hourly cost by accelerator
- Cost per 1K requests
- Cost efficiency (requests per dollar)
- Resource utilization vs cost
- Waste detection (underutilized instances)
- Daily cost trends
- Cost vs performance analysis

**Use Cases**:
- Cost monitoring and budgeting
- Identifying optimization opportunities
- Resource rightsizing decisions
- ROI analysis

### 3. SLA Compliance Dashboard (`sla-compliance-dashboard.json`)
**Purpose**: Service Level Agreement monitoring and compliance tracking
**Key Metrics**:
- Gold/Silver tier SLA compliance rates
- SLA violations count
- Latency distribution by tier
- Compliance by model
- Latency heatmaps
- Success rates
- Error budget burn rates
- Alert status

**Use Cases**:
- SLA monitoring and reporting
- Performance troubleshooting
- Customer experience tracking
- Incident response

## Installation Instructions

### Method 1: Grafana UI Import
1. Open Grafana web interface
2. Navigate to **Dashboards** → **Import**
3. Upload the JSON file or copy-paste the content
4. Configure data source (Prometheus)
5. Save the dashboard

### Method 2: Kubernetes ConfigMap (Automated)
```bash
# Create ConfigMap with dashboard definitions
kubectl create configmap grafana-dashboards \
  --from-file=dashboards/ \
  -n monitoring

# The dashboards will be automatically provisioned if Grafana is configured
# with the dashboard provider pointing to this ConfigMap
```

### Method 3: Grafana Provisioning
Add to your Grafana configuration:

```yaml
# grafana-dashboard-provider.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-provider
  namespace: monitoring
data:
  dashboards.yaml: |
    apiVersion: 1
    providers:
    - name: 'inference-platform'
      orgId: 1
      folder: 'ML Inference Platform'
      type: file
      disableDeletion: false
      updateIntervalSeconds: 10
      allowUiUpdates: true
      options:
        path: /var/lib/grafana/dashboards
```

## Dashboard Variables

All dashboards include template variables for filtering:

### Common Variables:
- **model**: Filter by specific model(s)
- **accelerator**: Filter by accelerator type (CPU/GPU/Inferentia)
- **time_range**: Adjustable time window
- **sla_tier**: Filter by SLA tier (Gold/Silver/Bronze)

### Usage:
- Variables appear at the top of each dashboard
- Support multi-select for comparing across different values
- Use "All" to see aggregate data

## Alerts Integration

Dashboards are integrated with Prometheus AlertManager:

### Annotations:
- **Deployments**: Shows model deployment events
- **SLA Violations**: Highlights when SLAs are breached
- **High Cost Alerts**: Marks cost threshold breaches
- **Scaling Events**: Shows auto-scaling activities

### Alert Rules:
Key alerts that appear on dashboards:
- `HighLatencyGoldTier`: P95 > 50ms for Gold tier
- `HighLatencySilverTier`: P95 > 150ms for Silver tier
- `HighErrorRate`: Error rate > 5%
- `HighQueueDepth`: Queue depth > 100
- `LowSLACompliance`: SLA compliance < target

## Customization

### Adding New Panels:
1. Edit the JSON file directly, or
2. Use Grafana UI to add panels and export JSON
3. Update the ConfigMap/file for persistence

### Custom Metrics:
To add custom metrics, ensure they're exposed by:
- Router service (`/metrics` endpoint)
- Backend services (`/health/metrics` endpoint)
- Custom metrics server
- Prometheus exporters

### Color Schemes:
- Green: Good/Healthy state
- Yellow: Warning/Attention needed
- Red: Critical/Action required
- Blue: Informational

## Troubleshooting

### Common Issues:

1. **No Data Showing**:
   - Verify Prometheus data source is configured
   - Check metric names match your deployment
   - Ensure time range includes data

2. **Missing Metrics**:
   - Verify services are exposing metrics
   - Check Prometheus scrape configuration
   - Validate metric labels match dashboard queries

3. **Performance Issues**:
   - Reduce time range for heavy queries
   - Use recording rules for complex calculations
   - Consider dashboard refresh intervals

### Metric Dependencies:
Dashboards expect these metrics to be available:
- `inference_requests_total`
- `inference_request_duration_seconds_bucket`
- `inference_queue_size`
- `inference_node_cost_per_hour`
- `inference_sla_violations_total`
- `container_cpu_usage_seconds_total`
- `container_memory_usage_bytes`

## Screenshots

Place dashboard screenshots in the `screenshots/` directory:
- `overview-dashboard.png`
- `cost-dashboard.png`
- `sla-dashboard.png`

## Support

For dashboard issues or feature requests:
1. Check the troubleshooting section above
2. Verify metric availability in Prometheus
3. Review Grafana logs for errors
4. Submit issues with dashboard JSON and error details
