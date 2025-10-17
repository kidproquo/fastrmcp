#!/bin/bash

# Generate version timestamp (YYYYMMDDHHMM format)
VERSION=$(date +%Y%m%d%H%M)

# Docker image details
IMAGE_NAME="ghcr.io/kidproquo/fastrmcp"
IMAGE_TAG="${IMAGE_NAME}:${VERSION}"
IMAGE_LATEST="${IMAGE_NAME}:latest"

echo "Building Docker image..."
echo "Version: ${VERSION}"
echo "Image tag: ${IMAGE_TAG}"

# Build the Docker image
docker build -t "${IMAGE_TAG}" -t "${IMAGE_LATEST}" .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo ""
    echo "Tagged as:"
    echo "  - ${IMAGE_TAG}"
    echo "  - ${IMAGE_LATEST}"
    echo ""
    echo "To push to GitHub Container Registry:"
    echo "  docker push ${IMAGE_TAG}"
    echo "  docker push ${IMAGE_LATEST}"
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi
