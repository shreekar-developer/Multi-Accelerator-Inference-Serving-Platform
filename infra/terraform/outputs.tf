output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_primary_security_group_id" {
  description = "Cluster security group that was created by Amazon EKS for the cluster"
  value       = module.eks.cluster_primary_security_group_id
}

output "vpc_id" {
  description = "ID of the VPC where the cluster is deployed"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "The CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "node_groups" {
  description = "EKS node groups"
  value       = module.eks.eks_managed_node_groups
  sensitive   = true
}

output "ecr_repositories" {
  description = "ECR repository URLs"
  value = {
    for k, v in aws_ecr_repository.repositories : k => v.repository_url
  }
}

output "dynamodb_table_name" {
  description = "Name of the DynamoDB table for model profiles"
  value       = aws_dynamodb_table.model_profiles.name
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "sqs_queue_url" {
  description = "URL of the SQS queue for async requests"
  value       = aws_sqs_queue.async_requests.url
}

output "kms_key_id" {
  description = "KMS key ID for encryption"
  value       = aws_kms_key.eks.key_id
}

output "router_irsa_role_arn" {
  description = "IAM role ARN for router service account"
  value       = module.irsa_router.iam_role_arn
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "aws_account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current.account_id
}

# Configuration values for applications
output "cluster_config" {
  description = "Cluster configuration for application deployment"
  value = {
    cluster_name = module.eks.cluster_name
    region       = var.aws_region
    account_id   = data.aws_caller_identity.current.account_id
    
    # Node groups
    node_groups = {
      cpu = {
        taint_key   = "accelerator"
        taint_value = "cpu"
        labels      = local.node_groups.cpu.labels
      }
      gpu = {
        taint_key   = "accelerator"
        taint_value = "gpu"
        labels      = local.node_groups.gpu.labels
      }
      inferentia = {
        taint_key   = "accelerator"
        taint_value = "inferentia"
        labels      = local.node_groups.inferentia.labels
      }
    }
    
    # Storage and messaging
    dynamodb_table = aws_dynamodb_table.model_profiles.name
    s3_bucket      = aws_s3_bucket.model_artifacts.bucket
    sqs_queue_url  = aws_sqs_queue.async_requests.url
    
    # Security
    kms_key_id         = aws_kms_key.eks.key_id
    router_role_arn    = module.irsa_router.iam_role_arn
    
    # Container registry
    ecr_repositories = {
      for k, v in aws_ecr_repository.repositories : k => v.repository_url
    }
  }
}

# Quick deployment commands
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "ecr_login_command" {
  description = "Command to login to ECR"
  value       = "aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
} 