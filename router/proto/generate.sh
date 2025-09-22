#!/bin/bash

# Generate Go code from protobuf definitions

set -e

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo "protoc is not installed. Please install Protocol Buffers compiler."
    echo "On macOS: brew install protobuf"
    echo "On Ubuntu: sudo apt-get install protobuf-compiler"
    exit 1
fi

# Check if protoc-gen-go is installed
if ! command -v protoc-gen-go &> /dev/null; then
    echo "Installing protoc-gen-go..."
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
fi

# Check if protoc-gen-go-grpc is installed
if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo "Installing protoc-gen-go-grpc..."
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
fi

# Create output directory
mkdir -p generated

echo "Generating Go code from protobuf definitions..."

# Generate Go code
protoc \
    --go_out=generated \
    --go_opt=paths=source_relative \
    --go-grpc_out=generated \
    --go-grpc_opt=paths=source_relative \
    --proto_path=. \
    inference.proto

echo "Generated files:"
find generated -name "*.go" -type f

echo "Proto generation completed successfully!"

# Copy generated files to the main proto directory
cp generated/*.go ./

echo "Copied generated files to proto directory"
