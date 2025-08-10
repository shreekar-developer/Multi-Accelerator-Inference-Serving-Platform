package routing

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/ml-serving-platform/router/internal/config"
	"github.com/ml-serving-platform/router/internal/metrics"
	"github.com/ml-serving-platform/router/internal/profiler"
)

// SLATier represents the service level agreement tier
type SLATier string

const (
	SLATierGold   SLATier = "gold"   // p99 ≤ 50ms
	SLATierSilver SLATier = "silver" // p99 ≤ 150ms
	SLATierBronze SLATier = "bronze" // Best effort
)

// AcceleratorType represents the type of accelerator
type AcceleratorType string

const (
	AcceleratorCPU        AcceleratorType = "cpu"
	AcceleratorGPU        AcceleratorType = "gpu"
	AcceleratorInferentia AcceleratorType = "inferentia"
)

// RoutingRequest represents an incoming inference request
type RoutingRequest struct {
	ModelID      string                 `json:"model_id"`
	Version      string                 `json:"version"`
	SLATier      SLATier               `json:"sla_tier"`
	Inputs       []map[string]interface{} `json:"inputs"`
	SessionID    string                 `json:"session_id,omitempty"`
	BatchSize    int                    `json:"batch_size,omitempty"`
	RequestID    string                 `json:"request_id"`
	Timestamp    time.Time              `json:"timestamp"`
}

// RoutingDecision represents the result of routing decision
type RoutingDecision struct {
	Accelerator       AcceleratorType `json:"accelerator"`
	BackendEndpoint   string          `json:"backend_endpoint"`
	EstimatedLatency  float64         `json:"estimated_latency_ms"`
	CostPerRequest    float64         `json:"cost_per_request"`
	ConfidenceScore   float64         `json:"confidence_score"`
	Reason            string          `json:"reason"`
	FallbackUsed      bool            `json:"fallback_used"`
	QueuePosition     int             `json:"queue_position,omitempty"`
}

// Router handles intelligent routing decisions
type Router struct {
	profiler         *profiler.Profiler
	metricsCollector *metrics.Collector
	config           *config.Config
	
	// Live metrics cache
	liveMetrics      sync.Map
	lastMetricsUpdate time.Time
	metricsMutex     sync.RWMutex
	
	// Sticky routing for sessions
	sessionRouting   sync.Map
	
	// Circuit breaker
	circuitBreaker   map[AcceleratorType]*CircuitBreaker
	cbMutex          sync.RWMutex
	
	// Cost table
	costTable        map[AcceleratorType]CostInfo
	costMutex        sync.RWMutex
}

// CircuitBreaker manages backend health
type CircuitBreaker struct {
	State        string    `json:"state"`
	FailureCount int       `json:"failure_count"`
	LastFailure  time.Time `json:"last_failure"`
	NextRetry    time.Time `json:"next_retry"`
}

// CostInfo holds cost information for an accelerator
type CostInfo struct {
	OnDemandCostPerHour  float64 `json:"on_demand_cost_per_hour"`
	SpotCostPerHour      float64 `json:"spot_cost_per_hour"`
	EffectiveQPS         float64 `json:"effective_qps"`
	UtilizationRate      float64 `json:"utilization_rate"`
}

// LiveMetrics holds real-time performance data
type LiveMetrics struct {
	P50Latency    float64 `json:"p50_latency_ms"`
	P95Latency    float64 `json:"p95_latency_ms"`
	P99Latency    float64 `json:"p99_latency_ms"`
	QueueDepth    int     `json:"queue_depth"`
	Utilization   float64 `json:"utilization_percent"`
	ThroughputQPS float64 `json:"throughput_qps"`
	ErrorRate     float64 `json:"error_rate_percent"`
	UpdatedAt     time.Time `json:"updated_at"`
}

// NewRouter creates a new router instance
func NewRouter(profiler *profiler.Profiler, metricsCollector *metrics.Collector, config *config.Config) *Router {
	router := &Router{
		profiler:         profiler,
		metricsCollector: metricsCollector,
		config:           config,
		circuitBreaker:   make(map[AcceleratorType]*CircuitBreaker),
		costTable:        initializeCostTable(),
	}
	
	// Initialize circuit breakers
	for _, accel := range []AcceleratorType{AcceleratorCPU, AcceleratorGPU, AcceleratorInferentia} {
		router.circuitBreaker[accel] = &CircuitBreaker{
			State: "closed",
		}
	}
	
	return router
}

// RouteRequest makes routing decision for an inference request
func (r *Router) RouteRequest(ctx context.Context, req *RoutingRequest) (*RoutingDecision, error) {
	if req.BatchSize == 0 {
		req.BatchSize = len(req.Inputs)
	}
	
	// Check for sticky routing (session affinity)
	if req.SessionID != "" {
		if decision, exists := r.getSessionRouting(req.SessionID); exists {
			// Validate the sticky route is still healthy
			if r.isAcceleratorHealthy(decision.Accelerator) {
				decision.Reason = "sticky_routing"
				return decision, nil
			}
			// Clear unhealthy sticky route
			r.clearSessionRouting(req.SessionID)
		}
	}
	
	// Get model profiles for all accelerators
	profiles, err := r.profiler.GetModelProfiles(ctx, req.ModelID, req.Version)
	if err != nil {
		return nil, fmt.Errorf("failed to get model profiles: %w", err)
	}
	
	if len(profiles) == 0 {
		return nil, errors.New("no model profiles found")
	}
	
	// Get live metrics
	liveMetrics := r.getLiveMetrics()
	
	// Filter candidates based on SLA requirements
	candidates := r.filterBySLA(profiles, req.SLATier, liveMetrics, req.BatchSize)
	
	if len(candidates) == 0 {
		// Use fallback strategy
		return r.fallbackRouting(profiles, liveMetrics, req)
	}
	
	// Rank candidates by cost-effectiveness
	decision := r.selectOptimalCandidate(candidates, liveMetrics, req)
	
	// Store sticky routing if session ID provided
	if req.SessionID != "" {
		r.setSessionRouting(req.SessionID, decision)
	}
	
	return decision, nil
}

// filterBySLA filters accelerators that can meet the SLA requirements
func (r *Router) filterBySLA(profiles map[AcceleratorType]*profiler.ModelProfile, 
	sla SLATier, liveMetrics map[AcceleratorType]*LiveMetrics, batchSize int) []*RoutingCandidate {
	
	slaLatencyBound := r.getSLALatencyBound(sla)
	candidates := make([]*RoutingCandidate, 0)
	
	for accelType, profile := range profiles {
		// Skip if circuit breaker is open
		if !r.isAcceleratorHealthy(accelType) {
			continue
		}
		
		// Get adjusted latency considering current load
		adjustedLatency := r.calculateAdjustedLatency(profile, liveMetrics[accelType], batchSize)
		
		// Check if it meets SLA
		if adjustedLatency <= slaLatencyBound {
			candidate := &RoutingCandidate{
				Accelerator:      accelType,
				Profile:          profile,
				AdjustedLatency:  adjustedLatency,
				CostPerRequest:   r.calculateCostPerRequest(accelType, liveMetrics[accelType]),
				ConfidenceScore:  r.calculateConfidenceScore(profile, liveMetrics[accelType]),
				QueuePosition:    r.getQueuePosition(accelType, liveMetrics),
			}
			candidates = append(candidates, candidate)
		}
	}
	
	return candidates
}

// RoutingCandidate represents a potential routing target
type RoutingCandidate struct {
	Accelerator     AcceleratorType
	Profile         *profiler.ModelProfile
	AdjustedLatency float64
	CostPerRequest  float64
	ConfidenceScore float64
	QueuePosition   int
}

// selectOptimalCandidate chooses the best candidate based on cost and performance
func (r *Router) selectOptimalCandidate(candidates []*RoutingCandidate, 
	liveMetrics map[AcceleratorType]*LiveMetrics, req *RoutingRequest) *RoutingDecision {
	
	// Sort by cost-effectiveness score
	sort.Slice(candidates, func(i, j int) bool {
		scoreI := r.calculateCostEffectivenessScore(candidates[i])
		scoreJ := r.calculateCostEffectivenessScore(candidates[j])
		return scoreI > scoreJ
	})
	
	best := candidates[0]
	
	return &RoutingDecision{
		Accelerator:      best.Accelerator,
		BackendEndpoint:  r.getBackendEndpoint(best.Accelerator),
		EstimatedLatency: best.AdjustedLatency,
		CostPerRequest:   best.CostPerRequest,
		ConfidenceScore:  best.ConfidenceScore,
		Reason:           "optimal_cost_performance",
		FallbackUsed:     false,
		QueuePosition:    best.QueuePosition,
	}
}

// calculateAdjustedLatency adjusts latency based on current load
func (r *Router) calculateAdjustedLatency(profile *profiler.ModelProfile, 
	live *LiveMetrics, batchSize int) float64 {
	
	baseLatency := profile.P95LatencyMs
	
	// Adjust for batch size
	if batchSize > 1 {
		batchFactor := math.Log2(float64(batchSize)) + 1
		baseLatency *= batchFactor
	}
	
	if live == nil {
		return baseLatency
	}
	
	// Adjust for current queue depth
	queuePenalty := float64(live.QueueDepth) * 2.0 // 2ms per queued request
	
	// Adjust for utilization
	utilizationPenalty := 0.0
	if live.Utilization > 80.0 {
		utilizationPenalty = (live.Utilization - 80.0) * 0.5
	}
	
	return baseLatency + queuePenalty + utilizationPenalty
}

// calculateCostPerRequest calculates the cost per request for an accelerator
func (r *Router) calculateCostPerRequest(accelType AcceleratorType, live *LiveMetrics) float64 {
	r.costMutex.RLock()
	costInfo, exists := r.costTable[accelType]
	r.costMutex.RUnlock()
	
	if !exists {
		return 0.0
	}
	
	effectiveQPS := costInfo.EffectiveQPS
	if live != nil && live.ThroughputQPS > 0 {
		effectiveQPS = live.ThroughputQPS
	}
	
	if effectiveQPS == 0 {
		return math.Inf(1)
	}
	
	// Use spot pricing if available, otherwise on-demand
	costPerHour := costInfo.SpotCostPerHour
	if costPerHour == 0 {
		costPerHour = costInfo.OnDemandCostPerHour
	}
	
	return (costPerHour / effectiveQPS) * (1000.0 / 3600.0) // Cost per 1k requests
}

// calculateConfidenceScore calculates confidence in the routing decision
func (r *Router) calculateConfidenceScore(profile *profiler.ModelProfile, live *LiveMetrics) float64 {
	score := 1.0
	
	// Reduce confidence if profile is old
	age := time.Since(profile.LastUpdated)
	if age > 24*time.Hour {
		score *= 0.8
	}
	
	// Reduce confidence if live metrics are stale
	if live != nil {
		metricsAge := time.Since(live.UpdatedAt)
		if metricsAge > 5*time.Minute {
			score *= 0.9
		}
		
		// Reduce confidence if error rate is high
		if live.ErrorRate > 1.0 {
			score *= (1.0 - live.ErrorRate/100.0)
		}
	}
	
	return math.Max(score, 0.1)
}

// calculateCostEffectivenessScore calculates a combined score for ranking
func (r *Router) calculateCostEffectivenessScore(candidate *RoutingCandidate) float64 {
	// Lower latency is better (higher score)
	latencyScore := 1.0 / (1.0 + candidate.AdjustedLatency/100.0)
	
	// Lower cost is better (higher score)
	costScore := 1.0 / (1.0 + candidate.CostPerRequest)
	
	// Higher confidence is better
	confidenceScore := candidate.ConfidenceScore
	
	// Weighted combination
	return 0.4*latencyScore + 0.4*costScore + 0.2*confidenceScore
}

// getSLALatencyBound returns the latency bound for a given SLA tier
func (r *Router) getSLALatencyBound(sla SLATier) float64 {
	switch sla {
	case SLATierGold:
		return 50.0
	case SLATierSilver:
		return 150.0
	case SLATierBronze:
		return math.Inf(1)
	default:
		return 150.0
	}
}

// fallbackRouting implements fallback strategy when no accelerator meets SLA
func (r *Router) fallbackRouting(profiles map[AcceleratorType]*profiler.ModelProfile,
	liveMetrics map[AcceleratorType]*LiveMetrics, req *RoutingRequest) (*RoutingDecision, error) {
	
	// Fallback order: GPU → Inferentia → CPU
	fallbackOrder := []AcceleratorType{AcceleratorGPU, AcceleratorInferentia, AcceleratorCPU}
	
	for _, accelType := range fallbackOrder {
		if profile, exists := profiles[accelType]; exists && r.isAcceleratorHealthy(accelType) {
			adjustedLatency := r.calculateAdjustedLatency(profile, liveMetrics[accelType], req.BatchSize)
			
			return &RoutingDecision{
				Accelerator:      accelType,
				BackendEndpoint:  r.getBackendEndpoint(accelType),
				EstimatedLatency: adjustedLatency,
				CostPerRequest:   r.calculateCostPerRequest(accelType, liveMetrics[accelType]),
				ConfidenceScore:  0.5, // Lower confidence for fallback
				Reason:           "fallback_no_sla_compliance",
				FallbackUsed:     true,
				QueuePosition:    r.getQueuePosition(accelType, liveMetrics),
			}, nil
		}
	}
	
	return nil, errors.New("no healthy accelerators available")
}

// Helper methods
func (r *Router) isAcceleratorHealthy(accelType AcceleratorType) bool {
	r.cbMutex.RLock()
	cb, exists := r.circuitBreaker[accelType]
	r.cbMutex.RUnlock()
	
	if !exists {
		return true
	}
	
	return cb.State == "closed" || (cb.State == "half-open" && time.Now().After(cb.NextRetry))
}

func (r *Router) getBackendEndpoint(accelType AcceleratorType) string {
	switch accelType {
	case AcceleratorCPU:
		return "cpu-backend.default.svc.cluster.local:8080"
	case AcceleratorGPU:
		return "gpu-backend.default.svc.cluster.local:8080"
	case AcceleratorInferentia:
		return "neuron-backend.default.svc.cluster.local:8080"
	default:
		return ""
	}
}

func (r *Router) getQueuePosition(accelType AcceleratorType, liveMetrics map[AcceleratorType]*LiveMetrics) int {
	if live, exists := liveMetrics[accelType]; exists {
		return live.QueueDepth
	}
	return 0
}

func (r *Router) getLiveMetrics() map[AcceleratorType]*LiveMetrics {
	r.metricsMutex.RLock()
	defer r.metricsMutex.RUnlock()
	
	result := make(map[AcceleratorType]*LiveMetrics)
	r.liveMetrics.Range(func(key, value interface{}) bool {
		if accelType, ok := key.(AcceleratorType); ok {
			if metrics, ok := value.(*LiveMetrics); ok {
				result[accelType] = metrics
			}
		}
		return true
	})
	
	return result
}

func (r *Router) getSessionRouting(sessionID string) (*RoutingDecision, bool) {
	if decision, exists := r.sessionRouting.Load(sessionID); exists {
		return decision.(*RoutingDecision), true
	}
	return nil, false
}

func (r *Router) setSessionRouting(sessionID string, decision *RoutingDecision) {
	r.sessionRouting.Store(sessionID, decision)
}

func (r *Router) clearSessionRouting(sessionID string) {
	r.sessionRouting.Delete(sessionID)
}

// StartHealthChecks begins background health monitoring
func (r *Router) StartHealthChecks(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			r.updateLiveMetrics(ctx)
			r.updateCircuitBreakers(ctx)
		}
	}
}

func (r *Router) updateLiveMetrics(ctx context.Context) {
	metrics := r.metricsCollector.GetLatestMetrics()
	
	r.metricsMutex.Lock()
	r.lastMetricsUpdate = time.Now()
	r.metricsMutex.Unlock()
	
	for accelType, accelMetrics := range metrics {
		r.liveMetrics.Store(accelType, accelMetrics)
	}
}

func (r *Router) updateCircuitBreakers(ctx context.Context) {
	liveMetrics := r.getLiveMetrics()
	
	r.cbMutex.Lock()
	defer r.cbMutex.Unlock()
	
	for accelType, cb := range r.circuitBreaker {
		if metrics, exists := liveMetrics[accelType]; exists {
			// Open circuit breaker if error rate is too high
			if metrics.ErrorRate > 10.0 && cb.State == "closed" {
				cb.State = "open"
				cb.FailureCount++
				cb.LastFailure = time.Now()
				cb.NextRetry = time.Now().Add(1 * time.Minute)
				log.Printf("Circuit breaker opened for %s due to high error rate: %.2f%%", accelType, metrics.ErrorRate)
			}
			
			// Transition to half-open state
			if cb.State == "open" && time.Now().After(cb.NextRetry) {
				cb.State = "half-open"
				log.Printf("Circuit breaker half-open for %s", accelType)
			}
			
			// Close circuit breaker if error rate is acceptable
			if cb.State == "half-open" && metrics.ErrorRate < 2.0 {
				cb.State = "closed"
				cb.FailureCount = 0
				log.Printf("Circuit breaker closed for %s", accelType)
			}
		}
	}
}

func initializeCostTable() map[AcceleratorType]CostInfo {
	return map[AcceleratorType]CostInfo{
		AcceleratorCPU: {
			OnDemandCostPerHour: 0.0864,  // c7g.large
			SpotCostPerHour:     0.0259,
			EffectiveQPS:        100.0,
		},
		AcceleratorGPU: {
			OnDemandCostPerHour: 1.006,   // g5.xlarge
			SpotCostPerHour:     0.302,
			EffectiveQPS:        50.0,
		},
		AcceleratorInferentia: {
			OnDemandCostPerHour: 0.76,    // inf2.xlarge
			SpotCostPerHour:     0.0,     // Spot not available for Inferentia
			EffectiveQPS:        80.0,
		},
	}
} 