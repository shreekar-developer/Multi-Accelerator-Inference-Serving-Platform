package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config holds all configuration for the router service
type Config struct {
	// Server configuration
	HTTPPort int
	GRPCPort int
	
	// AWS configuration
	AWSRegion string
	
	// DynamoDB configuration
	DynamoDB DynamoDBConfig
	
	// S3 configuration
	S3 S3Config
	
	// SQS configuration
	SQS SQSConfig
	
	// Backend configuration
	Backends BackendConfig
	
	// Metrics configuration
	Metrics MetricsConfig
	
	// Routing configuration
	Routing RoutingConfig
	
	// Circuit breaker configuration
	CircuitBreaker CircuitBreakerConfig
	
	// Logging configuration
	Logging LoggingConfig
}

type DynamoDBConfig struct {
	TableName string
	Region    string
}

type S3Config struct {
	BucketName string
	Region     string
	KeyPrefix  string
}

type SQSConfig struct {
	QueueURL string
	Region   string
}

type BackendConfig struct {
	CPUEndpoint        string
	GPUEndpoint        string
	InferentiaEndpoint string
	TimeoutSeconds     int
	MaxRetries         int
}

type MetricsConfig struct {
	UpdateInterval     time.Duration
	RetentionPeriod    time.Duration
	PrometheusEndpoint string
}

type RoutingConfig struct {
	StickySessionTTL     time.Duration
	MaxConcurrentRequests int
	DefaultSLATier       string
	CostOptimizationEnabled bool
	FallbackEnabled      bool
	BatchingEnabled      bool
	MaxBatchSize         int
	BatchTimeoutMs       int
}

type CircuitBreakerConfig struct {
	FailureThreshold   int
	RecoveryTimeout    time.Duration
	HalfOpenMaxCalls   int
	ErrorRateThreshold float64
}

type LoggingConfig struct {
	Level           string
	Format          string
	RequestLogging  bool
	MetricsLogging  bool
}

// Load loads configuration from environment variables
func Load() (*Config, error) {
	config := &Config{
		HTTPPort: getEnvAsInt("HTTP_PORT", 8080),
		GRPCPort: getEnvAsInt("GRPC_PORT", 9090),
		AWSRegion: getEnvAsString("AWS_REGION", "us-west-2"),
		
		DynamoDB: DynamoDBConfig{
			TableName: getEnvAsString("DYNAMODB_TABLE_NAME", "ml-serving-platform-model-profiles"),
			Region:    getEnvAsString("DYNAMODB_REGION", "us-west-2"),
		},
		
		S3: S3Config{
			BucketName: getEnvAsString("S3_BUCKET_NAME", ""),
			Region:     getEnvAsString("S3_REGION", "us-west-2"),
			KeyPrefix:  getEnvAsString("S3_KEY_PREFIX", "models/"),
		},
		
		SQS: SQSConfig{
			QueueURL: getEnvAsString("SQS_QUEUE_URL", ""),
			Region:   getEnvAsString("SQS_REGION", "us-west-2"),
		},
		
		Backends: BackendConfig{
			CPUEndpoint:        getEnvAsString("CPU_BACKEND_ENDPOINT", "cpu-backend.default.svc.cluster.local:8080"),
			GPUEndpoint:        getEnvAsString("GPU_BACKEND_ENDPOINT", "gpu-backend.default.svc.cluster.local:8080"),
			InferentiaEndpoint: getEnvAsString("INFERENTIA_BACKEND_ENDPOINT", "neuron-backend.default.svc.cluster.local:8080"),
			TimeoutSeconds:     getEnvAsInt("BACKEND_TIMEOUT_SECONDS", 30),
			MaxRetries:         getEnvAsInt("BACKEND_MAX_RETRIES", 3),
		},
		
		Metrics: MetricsConfig{
			UpdateInterval:     getEnvAsDuration("METRICS_UPDATE_INTERVAL", 30*time.Second),
			RetentionPeriod:    getEnvAsDuration("METRICS_RETENTION_PERIOD", 24*time.Hour),
			PrometheusEndpoint: getEnvAsString("PROMETHEUS_ENDPOINT", "http://prometheus-server.monitoring.svc.cluster.local:80"),
		},
		
		Routing: RoutingConfig{
			StickySessionTTL:        getEnvAsDuration("STICKY_SESSION_TTL", 1*time.Hour),
			MaxConcurrentRequests:   getEnvAsInt("MAX_CONCURRENT_REQUESTS", 1000),
			DefaultSLATier:          getEnvAsString("DEFAULT_SLA_TIER", "silver"),
			CostOptimizationEnabled: getEnvAsBool("COST_OPTIMIZATION_ENABLED", true),
			FallbackEnabled:         getEnvAsBool("FALLBACK_ENABLED", true),
			BatchingEnabled:         getEnvAsBool("BATCHING_ENABLED", true),
			MaxBatchSize:            getEnvAsInt("MAX_BATCH_SIZE", 32),
			BatchTimeoutMs:          getEnvAsInt("BATCH_TIMEOUT_MS", 100),
		},
		
		CircuitBreaker: CircuitBreakerConfig{
			FailureThreshold:   getEnvAsInt("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 5),
			RecoveryTimeout:    getEnvAsDuration("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", 1*time.Minute),
			HalfOpenMaxCalls:   getEnvAsInt("CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS", 3),
			ErrorRateThreshold: getEnvAsFloat("CIRCUIT_BREAKER_ERROR_RATE_THRESHOLD", 10.0),
		},
		
		Logging: LoggingConfig{
			Level:          getEnvAsString("LOG_LEVEL", "info"),
			Format:         getEnvAsString("LOG_FORMAT", "json"),
			RequestLogging: getEnvAsBool("REQUEST_LOGGING", true),
			MetricsLogging: getEnvAsBool("METRICS_LOGGING", false),
		},
	}
	
	// Validate required configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}
	
	return config, nil
}

// Validate validates the configuration
func (c *Config) Validate() error {
	if c.DynamoDB.TableName == "" {
		return fmt.Errorf("DYNAMODB_TABLE_NAME is required")
	}
	
	if c.S3.BucketName == "" {
		return fmt.Errorf("S3_BUCKET_NAME is required")
	}
	
	if c.SQS.QueueURL == "" {
		return fmt.Errorf("SQS_QUEUE_URL is required")
	}
	
	if c.HTTPPort < 1 || c.HTTPPort > 65535 {
		return fmt.Errorf("HTTP_PORT must be between 1 and 65535")
	}
	
	if c.GRPCPort < 1 || c.GRPCPort > 65535 {
		return fmt.Errorf("GRPC_PORT must be between 1 and 65535")
	}
	
	if c.HTTPPort == c.GRPCPort {
		return fmt.Errorf("HTTP_PORT and GRPC_PORT cannot be the same")
	}
	
	validSLATiers := map[string]bool{"gold": true, "silver": true, "bronze": true}
	if !validSLATiers[c.Routing.DefaultSLATier] {
		return fmt.Errorf("DEFAULT_SLA_TIER must be one of: gold, silver, bronze")
	}
	
	validLogLevels := map[string]bool{"debug": true, "info": true, "warn": true, "error": true}
	if !validLogLevels[c.Logging.Level] {
		return fmt.Errorf("LOG_LEVEL must be one of: debug, info, warn, error")
	}
	
	if c.Routing.MaxBatchSize < 1 || c.Routing.MaxBatchSize > 256 {
		return fmt.Errorf("MAX_BATCH_SIZE must be between 1 and 256")
	}
	
	if c.CircuitBreaker.ErrorRateThreshold < 0 || c.CircuitBreaker.ErrorRateThreshold > 100 {
		return fmt.Errorf("CIRCUIT_BREAKER_ERROR_RATE_THRESHOLD must be between 0 and 100")
	}
	
	return nil
}

// Helper functions for environment variable parsing
func getEnvAsString(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}

func getEnvAsInt(key string, defaultValue int) int {
	if valueStr, exists := os.LookupEnv(key); exists {
		if value, err := strconv.Atoi(valueStr); err == nil {
			return value
		}
	}
	return defaultValue
}

func getEnvAsFloat(key string, defaultValue float64) float64 {
	if valueStr, exists := os.LookupEnv(key); exists {
		if value, err := strconv.ParseFloat(valueStr, 64); err == nil {
			return value
		}
	}
	return defaultValue
}

func getEnvAsBool(key string, defaultValue bool) bool {
	if valueStr, exists := os.LookupEnv(key); exists {
		if value, err := strconv.ParseBool(valueStr); err == nil {
			return value
		}
	}
	return defaultValue
}

func getEnvAsDuration(key string, defaultValue time.Duration) time.Duration {
	if valueStr, exists := os.LookupEnv(key); exists {
		if value, err := time.ParseDuration(valueStr); err == nil {
			return value
		}
	}
	return defaultValue
}

// GetBackendEndpoint returns the endpoint for a given accelerator type
func (c *Config) GetBackendEndpoint(acceleratorType string) string {
	switch acceleratorType {
	case "cpu":
		return c.Backends.CPUEndpoint
	case "gpu":
		return c.Backends.GPUEndpoint
	case "inferentia":
		return c.Backends.InferentiaEndpoint
	default:
		return ""
	}
}

// IsProductionMode returns true if running in production mode
func (c *Config) IsProductionMode() bool {
	return getEnvAsString("ENVIRONMENT", "development") == "production"
}

// GetMetricsLabels returns common metrics labels
func (c *Config) GetMetricsLabels() map[string]string {
	return map[string]string{
		"service":     "router",
		"version":     getEnvAsString("VERSION", "unknown"),
		"environment": getEnvAsString("ENVIRONMENT", "development"),
		"region":      c.AWSRegion,
	}
} 