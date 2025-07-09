#!/bin/bash

# 1. prepare the tilus-artifacts image
# 1.1 first check if the image exists locally, if not
# 1.2 check if the image exists in docker hub under yyding user
# 1.3 if the iamge does not exist, build the image from the Dockerfile in the current directory
if [[ "$(docker images -q tilus-artifacts:latest 2> /dev/null)" == "" ]]; then
    if [[ "$(docker images -q yyding/tilus-artifacts:latest 2> /dev/null)" == "" ]]; then
        echo "Image not found locally, pulling from Docker Hub..."
        docker pull yyding/tilus-artifacts:latest
    fi
    docker tag yyding/tilus-artifacts:latest tilus-artifacts:latest
fi

# 2. create a container from the image if there is no container based on the image
if [[ "$(docker ps -aq -f name=tilus-artifacts)" == "" ]]; then
    echo "Creating a new container from tilus-artifacts image..."
    mkdir -p ./cache
    mkdir -p ./results
    mkdir -p ./precompiled-results
    # only map precompiled-cache if it exists on the host
    if [[ -d ./precompiled-cache ]]; then
        docker run --gpus all \
          -v ./cache:/app/cache \
          -v ./precompiled-cache:/app/precompiled-cache \
          -v ./results:/app/results \
          -v ./precompiled-results:/app/precompiled-results \
          -e "HF_TOKEN=$HF_TOKEN" \
          -d --name tilus-artifacts tilus-artifacts:latest
    else
        docker run --gpus all \
          -v ./cache:/app/cache \
          -v ./results:/app/results \
          -v ./precompiled-results:/app/precompiled-results \
          -e "HF_TOKEN=$HF_TOKEN" \
          -d --name tilus-artifacts tilus-artifacts:latest
    fi
fi

# 3. launch the container if it is not running
if [[ "$(docker ps -q -f name=tilus-artifacts)" == "" ]]; then
    echo "Starting the tilus-artifacts container..."
    docker start tilus-artifacts
fi

# 4. execute the entry point script in the container with the provided arguments
docker exec -it tilus-artifacts docker-entrypoint.sh python entry.py "$@"
