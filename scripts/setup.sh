#!/bin/bash
set -e

# Setup script for Multi-Accelerator Inference Serving Platform
# This script validates environment and installs necessary tools

echo "🚀 Setting up Multi-Accelerator Inference Serving Platform"
echo "============================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS environment"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew is not installed. Please install it first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    print_status "Homebrew is installed"
fi

# Check required tools
echo ""
echo "🔧 Checking required tools..."

# AWS CLI
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Install with: brew install awscli"
    else
        echo "  Install from: https://aws.amazon.com/cli/"
    fi
    exit 1
fi
print_status "AWS CLI is installed ($(aws --version | cut -d' ' -f1))"

# Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    else
        echo "  Install from: https://docs.docker.com/get-docker/"
    fi
    exit 1
fi
print_status "Docker is installed ($(docker --version | cut -d' ' -f3 | tr -d ','))"

# kubectl
if ! command -v kubectl &> /dev/null; then
    print_warning "kubectl is not installed - installing now..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install kubectl
    else
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    fi
fi
print_status "kubectl is installed ($(kubectl version --client --short 2>/dev/null | cut -d' ' -f3))"

# Terraform
if ! command -v terraform &> /dev/null; then
    print_warning "Terraform is not installed - installing now..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew tap hashicorp/tap
        brew install hashicorp/tap/terraform
    else
        wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
        sudo apt update && sudo apt install terraform
    fi
fi
print_status "Terraform is installed ($(terraform version | head -n1 | cut -d' ' -f2))"

# Istioctl
if ! command -v istioctl &> /dev/null; then
    print_warning "istioctl is not installed - installing now..."
    curl -L https://istio.io/downloadIstio | sh -
    sudo mv istio-*/bin/istioctl /usr/local/bin/
    rm -rf istio-*
fi
print_status "istioctl is installed ($(istioctl version --client --short 2>/dev/null || echo 'installed'))"

# Check AWS credentials
echo ""
echo "🔐 Checking AWS credentials..."
if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    AWS_REGION=$(aws configure get region || echo "us-west-2")
    print_status "AWS credentials are configured"
    echo "    Account ID: $ACCOUNT_ID"
    echo "    Region: $AWS_REGION"
else
    print_error "AWS credentials are not configured"
    echo "  Run: aws configure"
    exit 1
fi

# Validate Docker is running
echo ""
echo "🐳 Checking Docker daemon..."
if docker info &> /dev/null; then
    print_status "Docker daemon is running"
else
    print_error "Docker daemon is not running"
    echo "  Please start Docker Desktop or Docker daemon"
    exit 1
fi

# Check available disk space
echo ""
echo "💾 Checking disk space..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    AVAILABLE_GB=$(df -h . | awk 'NR==2 {print $4}' | sed 's/Gi//')
else
    AVAILABLE_GB=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
fi

if [[ ${AVAILABLE_GB%.*} -lt 10 ]]; then
    print_warning "Low disk space detected: ${AVAILABLE_GB}GB available"
    echo "  Recommend at least 10GB free space for building images"
else
    print_status "Sufficient disk space: ${AVAILABLE_GB}GB available"
fi

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p benchmarks/results
mkdir -p logs
mkdir -p configs
print_status "Project directories created"

# Make scripts executable
echo ""
echo "🔧 Setting up scripts..."
chmod +x scripts/*.sh
print_status "Scripts made executable"

# Validate Terraform configuration
echo ""
echo "🏗️  Validating Terraform configuration..."
cd infra/terraform
if terraform validate &> /dev/null; then
    print_status "Terraform configuration is valid"
else
    print_warning "Terraform configuration has issues - run 'terraform validate' for details"
fi
cd ../..

# Generate example configuration
echo ""
echo "⚙️  Generating example configuration..."
cat > configs/example.env << EOF
# AWS Configuration
AWS_REGION=us-west-2
AWS_ACCOUNT_ID=$ACCOUNT_ID

# Cluster Configuration
CLUSTER_NAME=ml-serving-platform
ENVIRONMENT=development

# Docker Registry
ECR_REPO_PREFIX=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com

# Build Configuration
VERSION=latest
BUILD_PARALLEL=true

# Deployment Configuration
DEPLOY_MONITORING=true
DEPLOY_SAMPLES=true
EOF
print_status "Example configuration created at configs/example.env"

# Display next steps
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Review and customize configs/example.env"
echo "   2. Deploy infrastructure: make deploy-infra"
echo "   3. Build and push images: make build-images \$ECR_REPO_PREFIX"
echo "   4. Deploy platform: make deploy-k8s"
echo "   5. Deploy sample models: make deploy-models"
echo ""
echo "📖 For more information, see README.md"
echo ""
echo "🔍 Useful commands:"
echo "   • make help              - Show all available commands"
echo "   • make status           - Check platform status"
echo "   • make open-grafana     - Open monitoring dashboard"
echo "   • make benchmark        - Run performance benchmarks"
echo ""
print_status "Ready to deploy the Multi-Accelerator Inference Serving Platform!" 