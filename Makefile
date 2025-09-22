.PHONY: help deploy-all deploy-infra deploy-k8s deploy-models build-images benchmark test clean

# Default AWS region and cluster name
AWS_REGION ?= us-west-2
CLUSTER_NAME ?= ml-serving-platform
ECR_REPO_PREFIX ?= $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

# Get AWS account ID
AWS_ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text)

help: ## Show this help message
	@echo "Multi-Accelerator Inference Serving Platform"
	@echo "============================================="
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Infrastructure and deployment
deploy-all: deploy-infra deploy-k8s deploy-models ## Deploy complete platform
	@echo "✅ Platform deployment complete!"
	@echo "Access Grafana: kubectl port-forward -n monitoring svc/grafana 3000:80"

deploy-infra: ## Deploy AWS infrastructure via Terraform
	@echo "🏗️  Deploying infrastructure..."
	cd infra/terraform && terraform init && terraform plan -out=tfplan && terraform apply tfplan
	@echo "⏳ Waiting for EKS cluster to be ready..."
	aws eks update-kubeconfig --region $(AWS_REGION) --name $(CLUSTER_NAME)
	kubectl wait --for=condition=Ready nodes --all --timeout=600s

deploy-k8s: build-images ## Deploy Kubernetes manifests
	@echo "🚀 Deploying Kubernetes resources..."
	# Install Istio
	istioctl install --set values.defaultRevision=default -y
	kubectl label namespace default istio-injection=enabled --overwrite
	
	# Deploy monitoring stack
	kubectl apply -f k8s/monitoring/
	kubectl wait --for=condition=available --timeout=300s deployment/prometheus-server -n monitoring
	
	# Deploy platform components
	kubectl apply -f k8s/platform/
	kubectl wait --for=condition=available --timeout=300s deployment/router deployment/cpu-backend deployment/gpu-backend deployment/neuron-backend

build-images: ## Build and push Docker images
	@echo "🐳 Building Docker images..."
	./scripts/build-images.sh $(ECR_REPO_PREFIX)

deploy-models: ## Deploy sample models
	@echo "📦 Deploying sample models..."
	python model_build/deploy_models.py --registry $(ECR_REPO_PREFIX)

# Development and testing
benchmark: ## Run comprehensive benchmarks
	@echo "📊 Running benchmarks..."
	python profiler/run_benchmarks.py --output-dir benchmarks/results/$(shell date +%Y%m%d_%H%M%S)

test: ## Run unit and integration tests
	@echo "🧪 Running tests..."
	cd router && go test ./...
	python -m pytest profiler/tests/
	python -m pytest model_build/tests/

load-test: ## Run load testing
	@echo "⚡ Running load tests..."
	cd profiler && python load_test.py --duration 300 --concurrent-users 50

# Utilities
open-grafana: ## Open Grafana dashboard
	@echo "📊 Opening Grafana (admin/admin)..."
	kubectl port-forward -n monitoring svc/grafana 3000:80 &
	sleep 3 && open http://localhost:3000

logs: ## View router logs
	kubectl logs -f deployment/router -c router

status: ## Check platform status
	@echo "📈 Platform Status"
	@echo "=================="
	kubectl get pods -A | grep -E "(router|backend|prometheus|grafana)"
	@echo ""
	@echo "🎯 SLA Compliance (last 1h):"
	@curl -s "http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=sla_compliance_percentage" | jq -r '.data.result[0].value[1]' || echo "Grafana not accessible"

clean: ## Clean up resources
	@echo "🧹 Cleaning up..."
	kubectl delete -f k8s/ --ignore-not-found=true
	cd infra/terraform && terraform destroy -auto-approve

# Model management
update-model: ## Update a model (usage: make update-model MODEL=distilbert VERSION=v2)
	@echo "🔄 Updating model $(MODEL) to version $(VERSION)..."
	python model_build/update_model.py --model $(MODEL) --version $(VERSION)

rollback-model: ## Rollback a model (usage: make rollback-model MODEL=distilbert)
	@echo "⏪ Rolling back model $(MODEL)..."
	kubectl patch deployment $(MODEL)-deployment -p '{"spec":{"template":{"metadata":{"annotations":{"deployment.kubernetes.io/revision":"1"}}}}}'

# Monitoring and debugging
debug-routing: ## Debug routing decisions
	kubectl exec -it deployment/router -- curl localhost:8080/debug/routing

scale-backend: ## Scale backend (usage: make scale-backend BACKEND=gpu REPLICAS=5)
	kubectl scale deployment $(BACKEND)-backend --replicas=$(REPLICAS)

cost-report: ## Generate cost report
	python profiler/cost_analysis.py --timeframe 24h --output cost_report.html

# Setup and initialization
setup: ## Initial setup and validation
	@echo "🔧 Setting up development environment..."
	./scripts/setup.sh

validate: ## Validate configuration
	@echo "✅ Validating platform configuration..."
	cd infra/terraform && terraform validate
	kubeval k8s/**/*.yaml
	@echo "Configuration is valid!"

# Emergency procedures
emergency-scale-down: ## Emergency scale down all backends
	@echo "🚨 Emergency scale down..."
	kubectl scale deployment --all --replicas=1

circuit-breaker: ## Activate circuit breaker
	kubectl patch configmap router-config -p '{"data":{"circuit_breaker":"true"}}' 