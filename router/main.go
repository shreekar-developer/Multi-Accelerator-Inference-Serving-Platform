package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/prometheus"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/sdk/metric"
	"google.golang.org/grpc"

	"github.com/ml-serving-platform/router/internal/config"
	"github.com/ml-serving-platform/router/internal/metrics"
	"github.com/ml-serving-platform/router/internal/profiler"
	"github.com/ml-serving-platform/router/internal/routing"
	"github.com/ml-serving-platform/router/internal/server"
	pb "github.com/ml-serving-platform/router/proto"
)

func main() {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize OpenTelemetry metrics
	ctx := context.Background()
	if err := initMetrics(ctx); err != nil {
		log.Fatalf("Failed to initialize metrics: %v", err)
	}

	// Initialize metrics collector
	metricsCollector := metrics.NewCollector()

	// Initialize model profiler
	modelProfiler, err := profiler.NewProfiler(cfg.DynamoDB, cfg.S3)
	if err != nil {
		log.Fatalf("Failed to initialize profiler: %v", err)
	}

	// Initialize router
	router := routing.NewRouter(modelProfiler, metricsCollector, cfg)

	// Start gRPC server
	grpcServer := grpc.NewServer()
	routerService := server.NewRouterServer(router)
	pb.RegisterInferenceRouterServer(grpcServer, routerService)

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.GRPCPort))
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	go func() {
		log.Printf("Starting gRPC server on port %d", cfg.GRPCPort)
		if err := grpcServer.Serve(listener); err != nil {
			log.Fatalf("Failed to serve gRPC: %v", err)
		}
	}()

	// Start HTTP gateway and metrics server
	httpServer := setupHTTPServer(routerService, cfg.HTTPPort)
	go func() {
		log.Printf("Starting HTTP server on port %d", cfg.HTTPPort)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to serve HTTP: %v", err)
		}
	}()

	// Start metrics background tasks
	go metricsCollector.StartCollection(ctx)
	go router.StartHealthChecks(ctx)

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down servers...")

	// Shutdown HTTP server
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		log.Printf("HTTP server forced to shutdown: %v", err)
	}

	// Shutdown gRPC server
	grpcServer.GracefulStop()

	log.Println("Servers shutdown complete")
}

func initMetrics(ctx context.Context) error {
	exporter, err := prometheus.New()
	if err != nil {
		return fmt.Errorf("failed to create Prometheus exporter: %w", err)
	}

	provider := metric.NewMeterProvider(metric.WithReader(exporter))
	otel.SetMeterProvider(provider)

	return nil
}

func setupHTTPServer(routerService *server.RouterServer, port int) *http.Server {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())

	// Health check endpoint
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "healthy"})
	})

	// Readiness check endpoint
	r.GET("/ready", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ready"})
	})

	// Metrics endpoint
	r.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Debug endpoints
	debug := r.Group("/debug")
	{
		debug.GET("/routing", routerService.GetRoutingStatus)
		debug.GET("/models", routerService.GetModelProfiles)
		debug.GET("/metrics", routerService.GetLiveMetrics)
	}

	// Main inference endpoint
	api := r.Group("/api/v1")
	{
		api.POST("/predict", routerService.PredictHTTP)
		api.POST("/predict/async", routerService.PredictAsyncHTTP)
		api.GET("/models", routerService.ListModelsHTTP)
		api.GET("/models/:model/versions/:version/profile", routerService.GetModelProfileHTTP)
	}

	return &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: r,
	}
} 