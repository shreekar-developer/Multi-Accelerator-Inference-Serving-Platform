package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/ml-serving-platform/router/internal/routing"
)

// RouterServer implements the HTTP and gRPC server for the router
type RouterServer struct {
	router *routing.Router
}

// NewRouterServer creates a new router server instance
func NewRouterServer(router *routing.Router) *RouterServer {
	return &RouterServer{
		router: router,
	}
}

// PredictHTTP handles HTTP prediction requests
func (s *RouterServer) PredictHTTP(c *gin.Context) {
	var req PredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid request format",
			"details": err.Error(),
		})
		return
	}

	// Validate request
	if req.ModelID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model_id is required"})
		return
	}
	if req.Version == "" {
		req.Version = "latest"
	}
	if req.SLATier == "" {
		req.SLATier = "silver"
	}

	// Generate request ID if not provided
	if req.RequestID == "" {
		req.RequestID = uuid.New().String()
	}

	// Convert to routing request
	routingReq := &routing.RoutingRequest{
		ModelID:   req.ModelID,
		Version:   req.Version,
		SLATier:   routing.SLATier(req.SLATier),
		Inputs:    req.Inputs,
		SessionID: req.SessionID,
		BatchSize: len(req.Inputs),
		RequestID: req.RequestID,
		Timestamp: time.Now(),
	}

	// Make routing decision
	decision, err := s.router.RouteRequest(c.Request.Context(), routingReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":      "routing failed",
			"details":    err.Error(),
			"request_id": req.RequestID,
		})
		return
	}

	// Forward request to selected backend
	response, err := s.forwardToBackend(c.Request.Context(), decision, &req)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error":      "backend request failed",
			"details":    err.Error(),
			"request_id": req.RequestID,
			"backend":    decision.Accelerator,
		})
		return
	}

	// Return response with routing metadata
	c.JSON(http.StatusOK, PredictResponse{
		RequestID:        req.RequestID,
		Outputs:          response.Outputs,
		LatencyMs:        response.LatencyMs,
		Accelerator:      string(decision.Accelerator),
		EstimatedLatency: decision.EstimatedLatency,
		CostPerRequest:   decision.CostPerRequest,
		ConfidenceScore:  decision.ConfidenceScore,
		QueuePosition:    decision.QueuePosition,
		Reason:           decision.Reason,
		FallbackUsed:     decision.FallbackUsed,
	})
}

// PredictAsyncHTTP handles asynchronous prediction requests
func (s *RouterServer) PredictAsyncHTTP(c *gin.Context) {
	var req PredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid request format",
			"details": err.Error(),
		})
		return
	}

	// Generate request ID if not provided
	if req.RequestID == "" {
		req.RequestID = uuid.New().String()
	}

	// For async requests, use bronze tier for cost optimization
	if req.SLATier == "" {
		req.SLATier = "bronze"
	}

	// TODO: Implement async processing via SQS
	// For now, return immediate acceptance
	c.JSON(http.StatusAccepted, gin.H{
		"request_id": req.RequestID,
		"status":     "accepted",
		"message":    "request queued for async processing",
	})
}

// ListModelsHTTP returns available models
func (s *RouterServer) ListModelsHTTP(c *gin.Context) {
	// TODO: Implement model listing via profiler
	models := []gin.H{
		{
			"model_id":     "distilbert_sst2",
			"version":      "v1.0",
			"accelerators": []string{"cpu", "gpu", "inferentia"},
			"sla_tiers":    []string{"gold", "silver", "bronze"},
		},
		{
			"model_id":     "resnet50",
			"version":      "v2.1",
			"accelerators": []string{"cpu", "gpu"},
			"sla_tiers":    []string{"silver", "bronze"},
		},
	}

	c.JSON(http.StatusOK, gin.H{
		"models": models,
		"count":  len(models),
	})
}

// GetModelProfileHTTP returns model profile information
func (s *RouterServer) GetModelProfileHTTP(c *gin.Context) {
	modelID := c.Param("model")
	version := c.Param("version")
	accelerator := c.Query("accelerator")

	if modelID == "" || version == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model and version are required"})
		return
	}

	// TODO: Implement profile retrieval via profiler
	profile := gin.H{
		"model_id":         modelID,
		"version":          version,
		"accelerator":      accelerator,
		"p95_latency_ms":   45.2,
		"qps_sustained":    120.5,
		"memory_mb":        1024,
		"cost_per_1k_req":  0.25,
		"last_updated":     time.Now().Format(time.RFC3339),
	}

	c.JSON(http.StatusOK, profile)
}

// GetRoutingStatus returns current routing status and decisions
func (s *RouterServer) GetRoutingStatus(c *gin.Context) {
	status := gin.H{
		"timestamp": time.Now().Format(time.RFC3339),
		"accelerators": gin.H{
			"cpu": gin.H{
				"healthy":      true,
				"queue_depth":  2,
				"utilization":  75.5,
				"p95_latency":  120.0,
				"error_rate":   0.5,
			},
			"gpu": gin.H{
				"healthy":      true,
				"queue_depth":  1,
				"utilization":  85.2,
				"p95_latency":  25.0,
				"error_rate":   0.2,
			},
			"inferentia": gin.H{
				"healthy":      true,
				"queue_depth":  1,
				"utilization":  70.8,
				"p95_latency":  35.0,
				"error_rate":   0.3,
			},
		},
		"routing_decisions": gin.H{
			"total_requests":    1250,
			"cpu_percentage":    45.2,
			"gpu_percentage":    35.8,
			"inferentia_percentage": 19.0,
		},
		"sla_compliance": gin.H{
			"gold_tier":   99.2,
			"silver_tier": 99.8,
			"bronze_tier": 100.0,
		},
	}

	c.JSON(http.StatusOK, status)
}

// GetModelProfiles returns all available model profiles
func (s *RouterServer) GetModelProfiles(c *gin.Context) {
	// TODO: Implement via profiler
	profiles := []gin.H{
		{
			"model_id":       "distilbert_sst2",
			"version":        "v1.0",
			"accelerator":    "gpu",
			"p95_latency_ms": 25.0,
			"qps_sustained":  300.0,
		},
		{
			"model_id":       "distilbert_sst2",
			"version":        "v1.0",
			"accelerator":    "cpu",
			"p95_latency_ms": 120.0,
			"qps_sustained":  100.0,
		},
	}

	c.JSON(http.StatusOK, gin.H{
		"profiles": profiles,
		"count":    len(profiles),
	})
}

// GetLiveMetrics returns current live metrics
func (s *RouterServer) GetLiveMetrics(c *gin.Context) {
	// TODO: Implement via metrics collector
	metrics := gin.H{
		"timestamp": time.Now().Format(time.RFC3339),
		"cpu": gin.H{
			"p95_latency_ms": 120.0,
			"queue_depth":    2,
			"throughput_qps": 100.0,
			"error_rate":     0.5,
			"utilization":    75.5,
		},
		"gpu": gin.H{
			"p95_latency_ms": 25.0,
			"queue_depth":    1,
			"throughput_qps": 300.0,
			"error_rate":     0.2,
			"utilization":    85.2,
		},
		"inferentia": gin.H{
			"p95_latency_ms": 35.0,
			"queue_depth":    1,
			"throughput_qps": 200.0,
			"error_rate":     0.3,
			"utilization":    70.8,
		},
	}

	c.JSON(http.StatusOK, metrics)
}

// forwardToBackend forwards the request to the selected backend
func (s *RouterServer) forwardToBackend(ctx context.Context, decision *routing.RoutingDecision, req *PredictRequest) (*BackendResponse, error) {
	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	// Prepare request payload
	payload := map[string]interface{}{
		"model_id": req.ModelID,
		"version":  req.Version,
		"inputs":   req.Inputs,
	}

	// Convert to JSON
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Make request to backend
	start := time.Now()
	
	// For demo purposes, simulate backend response
	latency := decision.EstimatedLatency + (float64(time.Since(start).Nanoseconds()) / 1e6)
	
	// Simulate response based on model and accelerator
	outputs := s.simulateModelOutput(req.ModelID, req.Inputs)

	return &BackendResponse{
		Outputs:   outputs,
		LatencyMs: latency,
	}, nil
}

// simulateModelOutput simulates model outputs for demo purposes
func (s *RouterServer) simulateModelOutput(modelID string, inputs []map[string]interface{}) []map[string]interface{} {
	outputs := make([]map[string]interface{}, len(inputs))
	
	for i := range inputs {
		switch modelID {
		case "distilbert_sst2":
			outputs[i] = map[string]interface{}{
				"label": "POSITIVE",
				"score": 0.9234,
			}
		case "resnet50":
			outputs[i] = map[string]interface{}{
				"class":       "Golden Retriever",
				"confidence":  0.8765,
				"predictions": []string{"Golden Retriever", "Labrador", "Dog"},
			}
		default:
			outputs[i] = map[string]interface{}{
				"result": "processed",
				"score":  0.85,
			}
		}
	}
	
	return outputs
}

// Request/Response types
type PredictRequest struct {
	ModelID   string                   `json:"model_id" binding:"required"`
	Version   string                   `json:"version"`
	SLATier   string                   `json:"sla_tier"`
	Inputs    []map[string]interface{} `json:"inputs" binding:"required"`
	SessionID string                   `json:"session_id,omitempty"`
	RequestID string                   `json:"request_id,omitempty"`
}

type PredictResponse struct {
	RequestID        string                   `json:"request_id"`
	Outputs          []map[string]interface{} `json:"outputs"`
	LatencyMs        float64                  `json:"latency_ms"`
	Accelerator      string                   `json:"accelerator"`
	EstimatedLatency float64                  `json:"estimated_latency_ms"`
	CostPerRequest   float64                  `json:"cost_per_request"`
	ConfidenceScore  float64                  `json:"confidence_score"`
	QueuePosition    int                      `json:"queue_position,omitempty"`
	Reason           string                   `json:"reason"`
	FallbackUsed     bool                     `json:"fallback_used"`
}

type BackendResponse struct {
	Outputs   []map[string]interface{} `json:"outputs"`
	LatencyMs float64                  `json:"latency_ms"`
} 