package metrics

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/api"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"
	
	"github.com/ml-serving-platform/router/internal/routing"
)

// Collector gathers live metrics from backends and Prometheus
type Collector struct {
	prometheusClient v1.API
	httpClient       *http.Client
	
	// Cache for metrics
	metricsCache     sync.Map
	lastUpdate       time.Time
	updateMutex      sync.RWMutex
	
	// Configuration
	updateInterval   time.Duration
	prometheusURL    string
	backendEndpoints map[routing.AcceleratorType]string
}

// LiveMetrics represents real-time performance data for an accelerator
type LiveMetrics struct {
	AcceleratorType routing.AcceleratorType `json:"accelerator_type"`
	P50Latency      float64                 `json:"p50_latency_ms"`
	P95Latency      float64                 `json:"p95_latency_ms"`
	P99Latency      float64                 `json:"p99_latency_ms"`
	QueueDepth      int                     `json:"queue_depth"`
	Utilization     float64                 `json:"utilization_percent"`
	ThroughputQPS   float64                 `json:"throughput_qps"`
	ErrorRate       float64                 `json:"error_rate_percent"`
	ActiveRequests  int                     `json:"active_requests"`
	AvailableMemory int                     `json:"available_memory_mb"`
	UpdatedAt       time.Time               `json:"updated_at"`
}

// BackendHealthStatus represents health status of a backend
type BackendHealthStatus struct {
	Healthy       bool      `json:"healthy"`
	ResponseTime  float64   `json:"response_time_ms"`
	LastCheck     time.Time `json:"last_check"`
	ErrorMessage  string    `json:"error_message,omitempty"`
}

// NewCollector creates a new metrics collector
func NewCollector() *Collector {
	// Initialize Prometheus client (would use actual config)
	client, err := api.NewClient(api.Config{
		Address: "http://prometheus-server.monitoring.svc.cluster.local:80",
	})
	if err != nil {
		log.Printf("Failed to create Prometheus client: %v", err)
		// Continue without Prometheus for demo
	}
	
	var promAPI v1.API
	if client != nil {
		promAPI = v1.NewAPI(client)
	}
	
	return &Collector{
		prometheusClient: promAPI,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		updateInterval: 30 * time.Second,
		prometheusURL:  "http://prometheus-server.monitoring.svc.cluster.local:80",
		backendEndpoints: map[routing.AcceleratorType]string{
			routing.AcceleratorCPU:        "http://cpu-backend.default.svc.cluster.local:8080",
			routing.AcceleratorGPU:        "http://gpu-backend.default.svc.cluster.local:8080",
			routing.AcceleratorInferentia: "http://neuron-backend.default.svc.cluster.local:8080",
		},
	}
}

// StartCollection begins background metrics collection
func (c *Collector) StartCollection(ctx context.Context) {
	ticker := time.NewTicker(c.updateInterval)
	defer ticker.Stop()
	
	// Initial collection
	c.collectAllMetrics(ctx)
	
	for {
		select {
		case <-ctx.Done():
			log.Println("Stopping metrics collection")
			return
		case <-ticker.C:
			c.collectAllMetrics(ctx)
		}
	}
}

// collectAllMetrics collects metrics from all sources
func (c *Collector) collectAllMetrics(ctx context.Context) {
	var wg sync.WaitGroup
	
	// Collect from each accelerator type
	for accelType := range c.backendEndpoints {
		wg.Add(1)
		go func(accel routing.AcceleratorType) {
			defer wg.Done()
			c.collectAcceleratorMetrics(ctx, accel)
		}(accelType)
	}
	
	wg.Wait()
	
	c.updateMutex.Lock()
	c.lastUpdate = time.Now()
	c.updateMutex.Unlock()
}

// collectAcceleratorMetrics collects metrics for a specific accelerator
func (c *Collector) collectAcceleratorMetrics(ctx context.Context, accelType routing.AcceleratorType) {
	metrics := &LiveMetrics{
		AcceleratorType: accelType,
		UpdatedAt:       time.Now(),
	}
	
	// Collect from Prometheus if available
	if c.prometheusClient != nil {
		c.collectPrometheusMetrics(ctx, accelType, metrics)
	}
	
	// Collect from backend health endpoint
	c.collectBackendMetrics(ctx, accelType, metrics)
	
	// Store in cache
	c.metricsCache.Store(accelType, metrics)
	
	log.Printf("Updated metrics for %s: p95=%.2fms, qps=%.2f, queue=%d", 
		accelType, metrics.P95Latency, metrics.ThroughputQPS, metrics.QueueDepth)
}

// collectPrometheusMetrics collects metrics from Prometheus
func (c *Collector) collectPrometheusMetrics(ctx context.Context, accelType routing.AcceleratorType, metrics *LiveMetrics) {
	queries := map[string]string{
		"p50_latency": fmt.Sprintf(`histogram_quantile(0.50, rate(inference_latency_seconds_bucket{accelerator="%s"}[5m])) * 1000`, accelType),
		"p95_latency": fmt.Sprintf(`histogram_quantile(0.95, rate(inference_latency_seconds_bucket{accelerator="%s"}[5m])) * 1000`, accelType),
		"p99_latency": fmt.Sprintf(`histogram_quantile(0.99, rate(inference_latency_seconds_bucket{accelerator="%s"}[5m])) * 1000`, accelType),
		"throughput":  fmt.Sprintf(`rate(inference_requests_total{accelerator="%s"}[5m])`, accelType),
		"error_rate":  fmt.Sprintf(`rate(inference_errors_total{accelerator="%s"}[5m]) / rate(inference_requests_total{accelerator="%s"}[5m]) * 100`, accelType, accelType),
		"queue_depth": fmt.Sprintf(`inference_queue_depth{accelerator="%s"}`, accelType),
		"utilization": fmt.Sprintf(`inference_utilization_percent{accelerator="%s"}`, accelType),
	}
	
	for metricName, query := range queries {
		result, err := c.queryPrometheus(ctx, query)
		if err != nil {
			log.Printf("Failed to query %s for %s: %v", metricName, accelType, err)
			continue
		}
		
		if len(result) > 0 {
			value := float64(result[0].Value)
			switch metricName {
			case "p50_latency":
				metrics.P50Latency = value
			case "p95_latency":
				metrics.P95Latency = value
			case "p99_latency":
				metrics.P99Latency = value
			case "throughput":
				metrics.ThroughputQPS = value
			case "error_rate":
				metrics.ErrorRate = value
			case "queue_depth":
				metrics.QueueDepth = int(value)
			case "utilization":
				metrics.Utilization = value
			}
		}
	}
}

// collectBackendMetrics collects metrics directly from backend health endpoints
func (c *Collector) collectBackendMetrics(ctx context.Context, accelType routing.AcceleratorType, metrics *LiveMetrics) {
	endpoint, exists := c.backendEndpoints[accelType]
	if !exists {
		return
	}
	
	healthURL := fmt.Sprintf("%s/health/metrics", endpoint)
	
	req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
	if err != nil {
		log.Printf("Failed to create request for %s: %v", accelType, err)
		return
	}
	
	resp, err := c.httpClient.Do(req)
	if err != nil {
		log.Printf("Failed to get metrics from %s: %v", accelType, err)
		// Use fallback values
		c.useFallbackMetrics(accelType, metrics)
		return
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		log.Printf("Non-200 response from %s: %d", accelType, resp.StatusCode)
		c.useFallbackMetrics(accelType, metrics)
		return
	}
	
	var healthMetrics struct {
		P95Latency     float64 `json:"p95_latency_ms"`
		QueueDepth     int     `json:"queue_depth"`
		ActiveRequests int     `json:"active_requests"`
		ThroughputQPS  float64 `json:"throughput_qps"`
		ErrorRate      float64 `json:"error_rate_percent"`
		MemoryUsage    struct {
			Available int `json:"available_mb"`
		} `json:"memory"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&healthMetrics); err != nil {
		log.Printf("Failed to decode metrics from %s: %v", accelType, err)
		c.useFallbackMetrics(accelType, metrics)
		return
	}
	
	// Update metrics with backend data (if not already set by Prometheus)
	if metrics.P95Latency == 0 {
		metrics.P95Latency = healthMetrics.P95Latency
	}
	if metrics.QueueDepth == 0 {
		metrics.QueueDepth = healthMetrics.QueueDepth
	}
	if metrics.ThroughputQPS == 0 {
		metrics.ThroughputQPS = healthMetrics.ThroughputQPS
	}
	if metrics.ErrorRate == 0 {
		metrics.ErrorRate = healthMetrics.ErrorRate
	}
	
	metrics.ActiveRequests = healthMetrics.ActiveRequests
	metrics.AvailableMemory = healthMetrics.MemoryUsage.Available
}

// useFallbackMetrics provides simulated metrics when real data is unavailable
func (c *Collector) useFallbackMetrics(accelType routing.AcceleratorType, metrics *LiveMetrics) {
	// Use simulated values based on accelerator type
	switch accelType {
	case routing.AcceleratorCPU:
		metrics.P50Latency = 80.0
		metrics.P95Latency = 120.0
		metrics.P99Latency = 180.0
		metrics.ThroughputQPS = 100.0
		metrics.QueueDepth = 2
		metrics.Utilization = 75.0
		metrics.ErrorRate = 0.5
		
	case routing.AcceleratorGPU:
		metrics.P50Latency = 15.0
		metrics.P95Latency = 25.0
		metrics.P99Latency = 40.0
		metrics.ThroughputQPS = 300.0
		metrics.QueueDepth = 1
		metrics.Utilization = 85.0
		metrics.ErrorRate = 0.2
		
	case routing.AcceleratorInferentia:
		metrics.P50Latency = 25.0
		metrics.P95Latency = 35.0
		metrics.P99Latency = 50.0
		metrics.ThroughputQPS = 200.0
		metrics.QueueDepth = 1
		metrics.Utilization = 70.0
		metrics.ErrorRate = 0.3
	}
	
	metrics.ActiveRequests = metrics.QueueDepth + 3
	metrics.AvailableMemory = 4096
}

// queryPrometheus executes a Prometheus query
func (c *Collector) queryPrometheus(ctx context.Context, query string) (model.Vector, error) {
	if c.prometheusClient == nil {
		return nil, fmt.Errorf("prometheus client not available")
	}
	
	result, warnings, err := c.prometheusClient.Query(ctx, query, time.Now())
	if err != nil {
		return nil, fmt.Errorf("prometheus query failed: %w", err)
	}
	
	if len(warnings) > 0 {
		log.Printf("Prometheus query warnings: %v", warnings)
	}
	
	vector, ok := result.(model.Vector)
	if !ok {
		return nil, fmt.Errorf("unexpected result type: %T", result)
	}
	
	return vector, nil
}

// GetLatestMetrics returns the latest metrics for all accelerators
func (c *Collector) GetLatestMetrics() map[routing.AcceleratorType]*LiveMetrics {
	result := make(map[routing.AcceleratorType]*LiveMetrics)
	
	c.metricsCache.Range(func(key, value interface{}) bool {
		if accelType, ok := key.(routing.AcceleratorType); ok {
			if metrics, ok := value.(*LiveMetrics); ok {
				result[accelType] = metrics
			}
		}
		return true
	})
	
	return result
}

// GetMetrics returns metrics for a specific accelerator
func (c *Collector) GetMetrics(accelType routing.AcceleratorType) *LiveMetrics {
	if value, exists := c.metricsCache.Load(accelType); exists {
		if metrics, ok := value.(*LiveMetrics); ok {
			return metrics
		}
	}
	return nil
}

// CheckBackendHealth checks the health of a specific backend
func (c *Collector) CheckBackendHealth(ctx context.Context, accelType routing.AcceleratorType) *BackendHealthStatus {
	endpoint, exists := c.backendEndpoints[accelType]
	if !exists {
		return &BackendHealthStatus{
			Healthy:      false,
			LastCheck:    time.Now(),
			ErrorMessage: "backend endpoint not configured",
		}
	}
	
	start := time.Now()
	healthURL := fmt.Sprintf("%s/health", endpoint)
	
	req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
	if err != nil {
		return &BackendHealthStatus{
			Healthy:      false,
			LastCheck:    time.Now(),
			ErrorMessage: fmt.Sprintf("failed to create request: %v", err),
		}
	}
	
	resp, err := c.httpClient.Do(req)
	responseTime := time.Since(start).Seconds() * 1000 // Convert to milliseconds
	
	if err != nil {
		return &BackendHealthStatus{
			Healthy:      false,
			ResponseTime: responseTime,
			LastCheck:    time.Now(),
			ErrorMessage: fmt.Sprintf("request failed: %v", err),
		}
	}
	defer resp.Body.Close()
	
	healthy := resp.StatusCode == http.StatusOK
	errorMessage := ""
	if !healthy {
		errorMessage = fmt.Sprintf("non-200 status: %d", resp.StatusCode)
	}
	
	return &BackendHealthStatus{
		Healthy:      healthy,
		ResponseTime: responseTime,
		LastCheck:    time.Now(),
		ErrorMessage: errorMessage,
	}
}

// GetHealthStatus returns health status for all backends
func (c *Collector) GetHealthStatus(ctx context.Context) map[routing.AcceleratorType]*BackendHealthStatus {
	result := make(map[routing.AcceleratorType]*BackendHealthStatus)
	
	var wg sync.WaitGroup
	var mutex sync.Mutex
	
	for accelType := range c.backendEndpoints {
		wg.Add(1)
		go func(accel routing.AcceleratorType) {
			defer wg.Done()
			status := c.CheckBackendHealth(ctx, accel)
			
			mutex.Lock()
			result[accel] = status
			mutex.Unlock()
		}(accelType)
	}
	
	wg.Wait()
	return result
}

// GetMetricsAge returns how old the current metrics are
func (c *Collector) GetMetricsAge() time.Duration {
	c.updateMutex.RLock()
	defer c.updateMutex.RUnlock()
	return time.Since(c.lastUpdate)
}

// IsStale returns true if metrics are too old to be reliable
func (c *Collector) IsStale() bool {
	return c.GetMetricsAge() > 2*c.updateInterval
} 