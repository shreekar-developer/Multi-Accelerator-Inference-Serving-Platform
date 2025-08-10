terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  cluster_name = var.cluster_name
  
  node_groups = {
    cpu = {
      name           = "cpu-graviton"
      instance_types = ["c7g.large", "c7g.xlarge", "c7g.2xlarge"]
      capacity_type  = "SPOT"
      scaling_config = {
        desired_size = 2
        max_size     = 10
        min_size     = 1
      }
      taints = [{
        key    = "accelerator"
        value  = "cpu"
        effect = "NO_SCHEDULE"
      }]
      labels = {
        "accelerator"     = "cpu"
        "node.kubernetes.io/instance-type" = "graviton"
      }
    }
    
    gpu = {
      name           = "gpu-nvidia"
      instance_types = ["g5.xlarge", "g5.2xlarge", "g5.4xlarge"]
      capacity_type  = "ON_DEMAND"
      scaling_config = {
        desired_size = 1
        max_size     = 5
        min_size     = 0
      }
      taints = [{
        key    = "accelerator"
        value  = "gpu"
        effect = "NO_SCHEDULE"
      }]
      labels = {
        "accelerator" = "gpu"
        "nvidia.com/gpu" = "true"
      }
    }
    
    inferentia = {
      name           = "inferentia-neuron"
      instance_types = ["inf2.xlarge", "inf2.8xlarge"]
      capacity_type  = "ON_DEMAND"
      scaling_config = {
        desired_size = 1
        max_size     = 3
        min_size     = 0
      }
      taints = [{
        key    = "accelerator"
        value  = "inferentia"
        effect = "NO_SCHEDULE"
      }]
      labels = {
        "accelerator" = "inferentia"
        "aws.amazon.com/neuron" = "true"
      }
    }
  }

  tags = {
    Environment = var.environment
    Project     = "ml-serving-platform"
    ManagedBy   = "terraform"
  }
}

# VPC
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }

  tags = local.tags
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.cluster_name
  cluster_version = "1.28"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # Cluster access entry
  enable_cluster_creator_admin_permissions = true

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    for k, v in local.node_groups : k => {
      name           = v.name
      instance_types = v.instance_types
      capacity_type  = v.capacity_type
      
      min_size     = v.scaling_config.min_size
      max_size     = v.scaling_config.max_size
      desired_size = v.scaling_config.desired_size

      ami_type = k == "cpu" ? "AL2_ARM_64" : "AL2_x86_64"
      
      labels = v.labels
      taints = v.taints

      # Launch template configuration
      create_launch_template = true
      launch_template_name   = "${local.cluster_name}-${k}"
      
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 50
            volume_type          = "gp3"
            iops                 = 3000
            throughput           = 150
            encrypted            = true
            kms_key_id          = aws_kms_key.eks.arn
            delete_on_termination = true
          }
        }
      }

      metadata_options = {
        http_endpoint               = "enabled"
        http_tokens                = "required"
        http_put_response_hop_limit = 2
        instance_metadata_tags      = "disabled"
      }

      tags = merge(local.tags, {
        "karpenter.sh/discovery" = local.cluster_name
      })
    }
  }

  tags = local.tags
}

# KMS key for EKS
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = local.tags
}

resource "aws_kms_alias" "eks" {
  name          = "alias/eks-${local.cluster_name}"
  target_key_id = aws_kms_key.eks.key_id
}

# ECR repositories for our Docker images
resource "aws_ecr_repository" "repositories" {
  for_each = toset([
    "router",
    "cpu-backend",
    "gpu-backend", 
    "neuron-backend",
    "profiler"
  ])

  name                 = "${local.cluster_name}/${each.key}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key        = aws_kms_key.eks.arn
  }

  tags = local.tags
}

# ECR lifecycle policies
resource "aws_ecr_lifecycle_policy" "repositories" {
  for_each   = aws_ecr_repository.repositories
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# DynamoDB table for model profiles
resource "aws_dynamodb_table" "model_profiles" {
  name           = "${local.cluster_name}-model-profiles"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "model_version"
  range_key      = "accelerator_config"

  attribute {
    name = "model_version"
    type = "S"
  }

  attribute {
    name = "accelerator_config"
    type = "S"
  }

  server_side_encryption {
    enabled     = true
    kms_key_id = aws_kms_key.eks.arn
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = local.tags
}

# S3 bucket for model artifacts
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${local.cluster_name}-model-artifacts-${random_string.bucket_suffix.result}"

  tags = local.tags
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.eks.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# SQS queue for async processing
resource "aws_sqs_queue" "async_requests" {
  name                      = "${local.cluster_name}-async-requests"
  delay_seconds             = 0
  max_message_size          = 2048
  message_retention_seconds = 1209600
  receive_wait_time_seconds = 10

  kms_master_key_id = aws_kms_key.eks.arn

  tags = local.tags
}

# IAM role for service accounts (IRSA)
module "irsa_router" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${local.cluster_name}-router-irsa"

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["default:router"]
    }
  }

  role_policy_arns = {
    dynamodb = aws_iam_policy.router_dynamodb.arn
    s3       = aws_iam_policy.router_s3.arn
    sqs      = aws_iam_policy.router_sqs.arn
  }

  tags = local.tags
}

# IAM policies
resource "aws_iam_policy" "router_dynamodb" {
  name        = "${local.cluster_name}-router-dynamodb"
  description = "IAM policy for router to access DynamoDB"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:Query",
          "dynamodb:Scan",
          "dynamodb:UpdateItem"
        ]
        Resource = aws_dynamodb_table.model_profiles.arn
      }
    ]
  })
}

resource "aws_iam_policy" "router_s3" {
  name        = "${local.cluster_name}-router-s3"
  description = "IAM policy for router to access S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.model_artifacts.arn}/*"
      }
    ]
  })
}

resource "aws_iam_policy" "router_sqs" {
  name        = "${local.cluster_name}-router-sqs"
  description = "IAM policy for router to access SQS"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage"
        ]
        Resource = aws_sqs_queue.async_requests.arn
      }
    ]
  })
} 