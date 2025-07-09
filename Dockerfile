# Start from the NVIDIA CUDA development image
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Set environment variable to avoid interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: git, cmake, vim, and wget (for Miniconda download)
# Using --no-install-recommends to keep the image size smaller
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    vim \
    wget \
    fonts-liberation && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
WORKDIR /tmp
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add Conda to the PATH for subsequent RUN commands during build
# Also, initialize Conda for bash and set the 'titus-artifacts' environment
# to activate automatically when a new bash session starts (due to ENTRYPOINT).
ENV PATH="/opt/conda/bin:$PATH"
RUN conda init bash && \
    echo "conda activate titus-artifacts" >> ~/.bashrc

# Create a Conda environment named 'titus-artifacts' with Python 3.10
# We need to explicitly source the conda initialization script and activate the base
# environment before creating the new environment in a RUN command.
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda create -n titus-artifacts python=3.10 -y"

# Set the working directory for your application code
WORKDIR /app

# Install baselines: vllm, triton, bitblas
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate titus-artifacts && \
    pip install --upgrade pip && \
    pip install pandas nvtx && \
    pip install vllm==0.7.3 bitblas==v0.0.1.dev15 && \
    pip install triton==3.1.0"

# Install baseline QuantLLM
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate titus-artifacts && \
    mkdir -p /app/artifacts && \
    rm -rf /app/artifacts/fp6_llm && \
    git clone https://github.com/usyd-fsalab/fp6_llm.git /app/artifacts/fp6_llm && \
    cd /app/artifacts/fp6_llm && \
    git checkout -b used-version 9802c5a && \
    pip install . && \
    cd /app && \
    rm -rf /app/artifacts/fp6_llm"

# Install baseline Marlin
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate titus-artifacts && \
    mkdir -p /app/artifacts && \
    rm -rf /app/artifacts/marlin && \
    git clone https://github.com/IST-DASLab/marlin.git /app/artifacts/marlin && \
    cd /app/artifacts/marlin && \
    git checkout -b used-version 1f25790 && \
    export TORCH_CUDA_ARCH_LIST='8.0+PTX' && \
    pip install . && \
    cd /app && \
    rm -rf /app/artifacts/marlin"

# Install dependencies for experiments: matplotlib
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate titus-artifacts && \
    pip install --upgrade pip && \
    pip install matplotlib==3.10.1 git-python hip-python-fork cuda-python==12.6.2.post1"

# Copy hidet source code into the container
COPY ./hidet /app/hidet

# Install hidet
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate titus-artifacts && \
    cd /app/hidet && \
    mkdir ./build && \
    cd ./build && \
    cmake .. && \
    make -j4 && \
    cd .. && \
    pip install -e ."

# Copy hidet source code into the container
COPY ./python /app/python
COPY ./setup.py /app/setup.py

# Install tilus
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate titus-artifacts && \
    cd /app && \
    pip install ."

# Copy the artifacts scripts
COPY ./artifacts /app/artifacts
COPY ./entry.py /app/entry.py
COPY ./precompiled-cache/bitblas /app/precompiled-cache/bitblas
COPY ./precompiled-cache/triton /app/precompiled-cache/triton
COPY ./precompiled-cache/mutis /app/precompiled-cache/mutis

# Set the entrypoint to run `docker-entrypoint.sh` script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["sleep", "infinity"]
