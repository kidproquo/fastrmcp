#!/bin/bash

# Docker image details
IMAGE_NAME="ghcr.io/kidproquo/fastrmcp:latest"
CONTAINER_NAME="fastrmcp"

# Create local directory for generated plots if it doesn't exist
PLOTS_DIR="$(pwd)/generated_plots"
mkdir -p "${PLOTS_DIR}"

echo "Starting FastRMCP Docker container..."
echo "Image: ${IMAGE_NAME}"
echo "Container name: ${CONTAINER_NAME}"
echo "Plots directory: ${PLOTS_DIR}"
echo "Port: 3003"

# Stop and remove existing container if it exists
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Stopping existing container..."
    docker stop ${CONTAINER_NAME} 2>/dev/null
    docker rm ${CONTAINER_NAME} 2>/dev/null
fi

# Run the container
docker run -d \
    --name ${CONTAINER_NAME} \
    -p 3003:3003 \
    -v "${PLOTS_DIR}:/app/rmcp_fastmcp/generated_plots" \
    ${IMAGE_NAME}

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container started successfully!"
    echo ""
    echo "Server URL: http://localhost:3003/mcp"
    echo "Generated plots will be saved to: ${PLOTS_DIR}"
    echo ""
    echo "Useful commands:"
    echo "  View logs:        docker logs -f ${CONTAINER_NAME}"
    echo "  Stop container:   docker stop ${CONTAINER_NAME}"
    echo "  Remove container: docker rm ${CONTAINER_NAME}"
else
    echo ""
    echo "❌ Failed to start container!"
    exit 1
fi
