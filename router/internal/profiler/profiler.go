package profiler

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	
	"github.com/ml-serving-platform/router/internal/routing"
)

// ModelProfile represents performance characteristics of a model on a specific accelerator
type ModelProfile struct {
	ModelID        string                      `json:"model_id" dynamodbav:"model_id"`
	Version        string                      `json:"version" dynamodbav:"version"`
	Accelerator    routing.AcceleratorType     `json:"accelerator" dynamodbav:"accelerator"`
	
	// Performance metrics
	P50LatencyMs   float64                     `json:"p50_latency_ms" dynamodbav:"p50_latency_ms"`
	P95LatencyMs   float64                     `json:"p95_latency_ms" dynamodbav:"p95_latency_ms"`
	P99LatencyMs   float64                     `json:"p99_latency_ms" dynamodbav:"p99_latency_ms"`
	QPSSustained   float64                     `json:"qps_sustained" dynamodbav:"qps_sustained"`
	
	// Resource usage
	MemoryMB       int                         `json:"memory_mb" dynamodbav:"memory_mb"`
	CPUUtilization float64                     `json:"cpu_utilization" dynamodbav:"cpu_utilization"`
	GPUUtilization float64                     `json:"gpu_utilization,omitempty" dynamodbav:"gpu_utilization,omitempty"`
	
	// Configuration
	BatchSize      int                         `json:"batch_size" dynamodbav:"batch_size"`
	SequenceLength int                         `json:"sequence_length,omitempty" dynamodbav:"sequence_length,omitempty"`
	Precision      string                      `json:"precision" dynamodbav:"precision"`
	
	// Compilation artifacts
	ArtifactS3Path string                      `json:"artifact_s3_path" dynamodbav:"artifact_s3_path"`
	CompileTime    time.Duration               `json:"compile_time" dynamodbav:"compile_time"`
	
	// Benchmarking metadata
	BenchmarkConfig BenchmarkConfig            `json:"benchmark_config" dynamodbav:"benchmark_config"`
	LastUpdated     time.Time                  `json:"last_updated" dynamodbav:"last_updated"`
	ProfileVersion  int                        `json:"profile_version" dynamodbav:"profile_version"`
	Notes          string                      `json:"notes,omitempty" dynamodbav:"notes,omitempty"`
	
	// Scaling characteristics
	ColdStartMs    float64                     `json:"cold_start_ms" dynamodbav:"cold_start_ms"`
	WarmupRequests int                         `json:"warmup_requests" dynamodbav:"warmup_requests"`
	MaxConcurrency int                         `json:"max_concurrency" dynamodbav:"max_concurrency"`
}

type BenchmarkConfig struct {
	TestDuration   time.Duration               `json:"test_duration" dynamodbav:"test_duration"`
	ConcurrencyLevels []int                    `json:"concurrency_levels" dynamodbav:"concurrency_levels"`
	BatchSizes     []int                       `json:"batch_sizes" dynamodbav:"batch_sizes"`
	SequenceLengths []int                      `json:"sequence_lengths,omitempty" dynamodbav:"sequence_lengths,omitempty"`
	InputDataPath  string                      `json:"input_data_path" dynamodbav:"input_data_path"`
	Environment    map[string]string           `json:"environment" dynamodbav:"environment"`
}

// Profiler handles model profile operations
type Profiler struct {
	dynamoClient *dynamodb.Client
	s3Client     *s3.Client
	tableName    string
	bucketName   string
}

// NewProfiler creates a new profiler instance
func NewProfiler(dynamoConfig, s3Config interface{}) (*Profiler, error) {
	ctx := context.Background()
	
	// Load AWS config
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS config: %w", err)
	}
	
	// Extract configuration (simplified for now)
	tableName := "ml-serving-platform-model-profiles"
	bucketName := "ml-serving-platform-model-artifacts"
	
	return &Profiler{
		dynamoClient: dynamodb.NewFromConfig(cfg),
		s3Client:     s3.NewFromConfig(cfg),
		tableName:    tableName,
		bucketName:   bucketName,
	}, nil
}

// GetModelProfiles retrieves all profiles for a model version
func (p *Profiler) GetModelProfiles(ctx context.Context, modelID, version string) (map[routing.AcceleratorType]*ModelProfile, error) {
	pk := fmt.Sprintf("%s#%s", modelID, version)
	
	input := &dynamodb.QueryInput{
		TableName:              aws.String(p.tableName),
		KeyConditionExpression: aws.String("model_version = :pk"),
		ExpressionAttributeValues: map[string]types.AttributeValue{
			":pk": &types.AttributeValueMemberS{Value: pk},
		},
	}
	
	result, err := p.dynamoClient.Query(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to query model profiles: %w", err)
	}
	
	profiles := make(map[routing.AcceleratorType]*ModelProfile)
	
	for _, item := range result.Items {
		var profile ModelProfile
		if err := attributevalue.UnmarshalMap(item, &profile); err != nil {
			log.Printf("Failed to unmarshal profile: %v", err)
			continue
		}
		profiles[profile.Accelerator] = &profile
	}
	
	return profiles, nil
}

// GetModelProfile retrieves a specific model profile
func (p *Profiler) GetModelProfile(ctx context.Context, modelID, version string, accelerator routing.AcceleratorType) (*ModelProfile, error) {
	pk := fmt.Sprintf("%s#%s", modelID, version)
	sk := fmt.Sprintf("%s#1#512", accelerator) // Default batch size and sequence length
	
	input := &dynamodb.GetItemInput{
		TableName: aws.String(p.tableName),
		Key: map[string]types.AttributeValue{
			"model_version":      &types.AttributeValueMemberS{Value: pk},
			"accelerator_config": &types.AttributeValueMemberS{Value: sk},
		},
	}
	
	result, err := p.dynamoClient.GetItem(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to get model profile: %w", err)
	}
	
	if result.Item == nil {
		return nil, fmt.Errorf("model profile not found")
	}
	
	var profile ModelProfile
	if err := attributevalue.UnmarshalMap(result.Item, &profile); err != nil {
		return nil, fmt.Errorf("failed to unmarshal profile: %w", err)
	}
	
	return &profile, nil
}

// StoreModelProfile stores or updates a model profile
func (p *Profiler) StoreModelProfile(ctx context.Context, profile *ModelProfile) error {
	profile.LastUpdated = time.Now()
	profile.ProfileVersion++
	
	pk := fmt.Sprintf("%s#%s", profile.ModelID, profile.Version)
	sk := fmt.Sprintf("%s#%d#%d", profile.Accelerator, profile.BatchSize, profile.SequenceLength)
	
	// Marshal profile to DynamoDB format
	item, err := attributevalue.MarshalMap(profile)
	if err != nil {
		return fmt.Errorf("failed to marshal profile: %w", err)
	}
	
	// Set partition and sort keys
	item["model_version"] = &types.AttributeValueMemberS{Value: pk}
	item["accelerator_config"] = &types.AttributeValueMemberS{Value: sk}
	
	input := &dynamodb.PutItemInput{
		TableName: aws.String(p.tableName),
		Item:      item,
	}
	
	_, err = p.dynamoClient.PutItem(ctx, input)
	if err != nil {
		return fmt.Errorf("failed to store model profile: %w", err)
	}
	
	log.Printf("Stored profile for %s/%s on %s", profile.ModelID, profile.Version, profile.Accelerator)
	return nil
}

// ListModels returns all available models
func (p *Profiler) ListModels(ctx context.Context) ([]ModelSummary, error) {
	input := &dynamodb.ScanInput{
		TableName:            aws.String(p.tableName),
		ProjectionExpression: aws.String("model_version, accelerator_config, last_updated"),
	}
	
	result, err := p.dynamoClient.Scan(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to scan models: %w", err)
	}
	
	modelMap := make(map[string]*ModelSummary)
	
	for _, item := range result.Items {
		modelVersion, ok := item["model_version"].(*types.AttributeValueMemberS)
		if !ok {
			continue
		}
		
		acceleratorConfig, ok := item["accelerator_config"].(*types.AttributeValueMemberS)
		if !ok {
			continue
		}
		
		lastUpdated, ok := item["last_updated"].(*types.AttributeValueMemberS)
		if !ok {
			continue
		}
		
		// Parse model ID and version
		parts := strings.Split(modelVersion.Value, "#")
		if len(parts) != 2 {
			continue
		}
		modelID, version := parts[0], parts[1]
		
		// Parse accelerator
		accelParts := strings.Split(acceleratorConfig.Value, "#")
		if len(accelParts) < 1 {
			continue
		}
		accelerator := accelParts[0]
		
		// Parse timestamp
		updatedTime, err := time.Parse(time.RFC3339, lastUpdated.Value)
		if err != nil {
			continue
		}
		
		key := fmt.Sprintf("%s#%s", modelID, version)
		if summary, exists := modelMap[key]; exists {
			summary.Accelerators = append(summary.Accelerators, accelerator)
			if updatedTime.After(summary.LastUpdated) {
				summary.LastUpdated = updatedTime
			}
		} else {
			modelMap[key] = &ModelSummary{
				ModelID:      modelID,
				Version:      version,
				Accelerators: []string{accelerator},
				LastUpdated:  updatedTime,
			}
		}
	}
	
	// Convert map to slice
	var models []ModelSummary
	for _, summary := range modelMap {
		models = append(models, *summary)
	}
	
	return models, nil
}

type ModelSummary struct {
	ModelID      string    `json:"model_id"`
	Version      string    `json:"version"`
	Accelerators []string  `json:"accelerators"`
	LastUpdated  time.Time `json:"last_updated"`
}

// BenchmarkModel runs comprehensive benchmarking for a model on an accelerator
func (p *Profiler) BenchmarkModel(ctx context.Context, modelID, version string, accelerator routing.AcceleratorType, config BenchmarkConfig) (*ModelProfile, error) {
	log.Printf("Starting benchmark for %s/%s on %s", modelID, version, accelerator)
	
	// Initialize profile
	profile := &ModelProfile{
		ModelID:         modelID,
		Version:         version,
		Accelerator:     accelerator,
		BenchmarkConfig: config,
		LastUpdated:     time.Now(),
		ProfileVersion:  1,
	}
	
	// Run benchmarks for different configurations
	for _, batchSize := range config.BatchSizes {
		for _, seqLen := range config.SequenceLengths {
			if seqLen == 0 {
				seqLen = 512 // Default sequence length
			}
			
			benchResult, err := p.runSingleBenchmark(ctx, modelID, version, accelerator, batchSize, seqLen, config)
			if err != nil {
				log.Printf("Benchmark failed for batch=%d, seq=%d: %v", batchSize, seqLen, err)
				continue
			}
			
			// Update profile with best results (lowest latency for default config)
			if batchSize == 1 && seqLen == 512 {
				profile.P50LatencyMs = benchResult.P50LatencyMs
				profile.P95LatencyMs = benchResult.P95LatencyMs
				profile.P99LatencyMs = benchResult.P99LatencyMs
				profile.QPSSustained = benchResult.QPSSustained
				profile.MemoryMB = benchResult.MemoryMB
				profile.CPUUtilization = benchResult.CPUUtilization
				profile.GPUUtilization = benchResult.GPUUtilization
			}
			
			// Store individual configuration profile
			configProfile := *profile
			configProfile.BatchSize = batchSize
			configProfile.SequenceLength = seqLen
			configProfile.P50LatencyMs = benchResult.P50LatencyMs
			configProfile.P95LatencyMs = benchResult.P95LatencyMs
			configProfile.P99LatencyMs = benchResult.P99LatencyMs
			configProfile.QPSSustained = benchResult.QPSSustained
			
			if err := p.StoreModelProfile(ctx, &configProfile); err != nil {
				log.Printf("Failed to store profile for batch=%d, seq=%d: %v", batchSize, seqLen, err)
			}
		}
	}
	
	// Estimate cold start and scaling characteristics
	profile.ColdStartMs = p.estimateColdStart(accelerator)
	profile.WarmupRequests = p.estimateWarmupRequests(accelerator)
	profile.MaxConcurrency = p.estimateMaxConcurrency(accelerator, profile.MemoryMB)
	
	log.Printf("Benchmark completed for %s/%s on %s", modelID, version, accelerator)
	return profile, nil
}

type BenchmarkResult struct {
	P50LatencyMs    float64
	P95LatencyMs    float64
	P99LatencyMs    float64
	QPSSustained    float64
	MemoryMB        int
	CPUUtilization  float64
	GPUUtilization  float64
}

// runSingleBenchmark runs a benchmark for a specific configuration
func (p *Profiler) runSingleBenchmark(ctx context.Context, modelID, version string, accelerator routing.AcceleratorType, batchSize, seqLen int, config BenchmarkConfig) (*BenchmarkResult, error) {
	// This would typically invoke the actual backend for benchmarking
	// For now, we'll simulate with realistic values based on accelerator type
	
	baseLatency := p.getBaseLatency(accelerator, batchSize)
	
	// Simulate latency distribution
	p50 := baseLatency
	p95 := baseLatency * 1.5
	p99 := baseLatency * 2.0
	
	// Simulate throughput
	qps := p.getBaseThroughput(accelerator) / float64(batchSize)
	
	// Simulate resource usage
	memory := p.getBaseMemory(accelerator, batchSize)
	cpuUtil := p.getBaseCPUUtilization(accelerator)
	gpuUtil := 0.0
	if accelerator == routing.AcceleratorGPU {
		gpuUtil = 85.0
	}
	
	return &BenchmarkResult{
		P50LatencyMs:   p50,
		P95LatencyMs:   p95,
		P99LatencyMs:   p99,
		QPSSustained:   qps,
		MemoryMB:       memory,
		CPUUtilization: cpuUtil,
		GPUUtilization: gpuUtil,
	}, nil
}

// Helper functions for simulation (would be replaced with actual benchmarking)
func (p *Profiler) getBaseLatency(accelerator routing.AcceleratorType, batchSize int) float64 {
	base := map[routing.AcceleratorType]float64{
		routing.AcceleratorCPU:        100.0,
		routing.AcceleratorGPU:        20.0,
		routing.AcceleratorInferentia: 35.0,
	}
	
	latency := base[accelerator]
	if batchSize > 1 {
		latency *= float64(batchSize) * 0.7 // Batching efficiency
	}
	
	return latency
}

func (p *Profiler) getBaseThroughput(accelerator routing.AcceleratorType) float64 {
	return map[routing.AcceleratorType]float64{
		routing.AcceleratorCPU:        100.0,
		routing.AcceleratorGPU:        300.0,
		routing.AcceleratorInferentia: 200.0,
	}[accelerator]
}

func (p *Profiler) getBaseMemory(accelerator routing.AcceleratorType, batchSize int) int {
	base := map[routing.AcceleratorType]int{
		routing.AcceleratorCPU:        512,
		routing.AcceleratorGPU:        2048,
		routing.AcceleratorInferentia: 1024,
	}
	
	return base[accelerator] + (batchSize-1)*100
}

func (p *Profiler) getBaseCPUUtilization(accelerator routing.AcceleratorType) float64 {
	return map[routing.AcceleratorType]float64{
		routing.AcceleratorCPU:        95.0,
		routing.AcceleratorGPU:        30.0,
		routing.AcceleratorInferentia: 60.0,
	}[accelerator]
}

func (p *Profiler) estimateColdStart(accelerator routing.AcceleratorType) float64 {
	return map[routing.AcceleratorType]float64{
		routing.AcceleratorCPU:        500.0,
		routing.AcceleratorGPU:        2000.0,
		routing.AcceleratorInferentia: 1500.0,
	}[accelerator]
}

func (p *Profiler) estimateWarmupRequests(accelerator routing.AcceleratorType) int {
	return map[routing.AcceleratorType]int{
		routing.AcceleratorCPU:        5,
		routing.AcceleratorGPU:        10,
		routing.AcceleratorInferentia: 8,
	}[accelerator]
}

func (p *Profiler) estimateMaxConcurrency(accelerator routing.AcceleratorType, memoryMB int) int {
	baseMemory := map[routing.AcceleratorType]int{
		routing.AcceleratorCPU:        8192,
		routing.AcceleratorGPU:        24576,
		routing.AcceleratorInferentia: 16384,
	}[accelerator]
	
	return baseMemory / memoryMB
}

// DeleteModelProfile removes a model profile
func (p *Profiler) DeleteModelProfile(ctx context.Context, modelID, version string, accelerator routing.AcceleratorType) error {
	pk := fmt.Sprintf("%s#%s", modelID, version)
	sk := fmt.Sprintf("%s#1#512", accelerator)
	
	input := &dynamodb.DeleteItemInput{
		TableName: aws.String(p.tableName),
		Key: map[string]types.AttributeValue{
			"model_version":      &types.AttributeValueMemberS{Value: pk},
			"accelerator_config": &types.AttributeValueMemberS{Value: sk},
		},
	}
	
	_, err := p.dynamoClient.DeleteItem(ctx, input)
	if err != nil {
		return fmt.Errorf("failed to delete model profile: %w", err)
	}
	
	return nil
}

// GetProfileMetrics returns aggregated metrics for monitoring
func (p *Profiler) GetProfileMetrics(ctx context.Context) (*ProfileMetrics, error) {
	// Scan table to get counts and statistics
	input := &dynamodb.ScanInput{
		TableName: aws.String(p.tableName),
		Select:    types.SelectCount,
	}
	
	result, err := p.dynamoClient.Scan(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to get profile metrics: %w", err)
	}
	
	return &ProfileMetrics{
		TotalProfiles:   int(*result.Count),
		LastUpdated:     time.Now(),
	}, nil
}

type ProfileMetrics struct {
	TotalProfiles int       `json:"total_profiles"`
	LastUpdated   time.Time `json:"last_updated"`
} 