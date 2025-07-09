# Tilus Artifacts

This repository contains the artifacts for the following paper:
```bibtex
Tilus: A Virtual Machine for Arbitrary Low-Precision GPGPU Computation in LLM Serving
Yaoyao Ding, Bohan Hou, Xiao Zhang, Allan Lin, Tianqi Chen, Cody Yu Hao, Yida Wang, Gennady Pekhimenko
```

We provide scripts to reproduce all the results in the evaluation section of the paper.

Requirements: 
- Ubuntu 22.04 or later
- NVIDIA GPU driver version 565 or later
- One NVIDIA GPU with ampere or later architecture (for functional correctness tests) or NVIDIA L40s (used in the paper).

## 1. Clone the artifacts repository
```bash
git clone https://github.com/yaoyaoding/tilus-artifacts.git tilus
```

## 2. Setup docker environment

### 2.1 Install docker and NVIDIA Container Toolkit

**Install Docker**

```bash
sudo apt-get update
sudo apt-get install -y docker.io
```

**Install NVIDIA Container Toolkit**

Install NVIDIA Container Toolkit by following the instructions in the official documentation:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

**Install NVIDIA Driver Installation**

If your system does not have the NVIDIA driver installed, you can follow the instructions in the official documentation:
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation

### 2.2 Pull the docker image
```bash
cd tilus
bash run.sh -h  # pull the docker image and show the help message for the artifacts
```

### 2.3 Lock the frequency

We recommend locking the GPU frequency to the TDP (Thermal design power) frequency for more stable evaluation. This
requires sudo privileges. You can use the following command to lock the frequency. This step is not necessary, but it
could produce more stable results and close to the results in the paper.

```bash
python scripts/lock-gpu-clocks.py --mode base # lock the GPU frequency to the base clock (TDP frequency)
```

After the evaluation, you can reset the GPU frequency via
```bash
python scripts/lock-gpu-clocks.py --mode reset 
```
which will dynamically adjust the GPU frequency based on the workload.

## 3. Run the experiments

### 3.1 Run with pre-compiled cache

To save the evaluation time, we provide the pre-compiled cache for Triton, Ladder, and Tilus (ours). Run the following
commands to perform the experiments with pre-compiled cache:

```bash
export HF_TOKEN=<your_huggingface_token>  # set your Hugging Face token
bash run.sh   # run all experiments with pre-compiled cache 
```

We recommend setting the `HF_TOKEN` environment variable to your Hugging Face token with read access to the following models:
- `google/gemma-2-9b`
- `meta-llama/Meta-Llama-3-70B-Instruct`

since the experiments will need to download these models from Hugging Face. When it's not set, the script will fail to
run the experiments on these models and only reproduce the results for
- `Qwen/Qwen2.5-Coder-32B-Instruct`

since it does not require a Hugging Face token to download.

After the command, there will be a `figureN.pdf` (where `N` is the figure number 9, 10, 11, and 13) under the `precompiled-results` 
directory with the evaluation results, similar to the figures in the paper. 

That's all you need to do to reproduce the results in the paper. The pre-compiled kernels used in the experiments for
Triton, Ladder (aka, bitblas), and Tilus (Ours) are stored in the `./precompiled-cache` directory **inside** the docker 
container. You can use ``./enter.sh`` to enter the docker container and check the pre-compiled cache.

### 3.2 (Optional) Run without pre-compiled cache

If you want to run the experiments without pre-compiled cache, or you are not using NVIDIA L40s, you can run the 
following command:

```bash
bash run.sh --no-cache  # run all experiments without pre-compiled cache
```
This will take 3 to 5 hours to run all experiments, as it will compile the Triton, Ladder, and Tilus kernels from scratch. 
The time heavily depends on the performance of the CPU (especially the number of cores).
After the completion, there will be a `figureN.pdf` (where `N` is the figure number) under the `results` directory with
the evaluation results, similar to the figures in the paper. The `cache` directory will be used to store the compiled 
kernels for Triton, Ladder, and Tilus (Ours).

## 4. Notes

Step 3 runs all the following experiments in the paper: figure 9, 10, 11, and 13. Running `bash run.sh` will take a long time
to complete. If you want to run a single experiment at a time, you can use the following command:

```bash
bash run.sh --figure <id>
```

Where `<id>` is one of the following: 
- `9`: figure 9 (operator experiment)
- `10`: figure 10 (coverage experiment)
- `11`: figure 11 (end-to-end experiment)
- `13`: figure 13 (batch size experiment)

We can get the results in Figure 12 by running the Figure 11 experiment on A100 and H100 GPUs, thus we do not provide a 
separate script for Figure 12.

## 5. FAQ

1. `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

This error indicates that the NVIDIA Container Toolkit is not installed or not configured correctly. Please follow the
instructions in the [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to install and configure it.

2. `401 Client Error: Unauthorized for url: https://huggingface.co/api/models/google/gemma-2-9b/tree/main?recursive=True&expand=False`

This error indicates that the Huggingface token is not set or does not have read access to the model. Please set the
token using the `HF_TOKEN` environment variable as described in the README. You can create a Huggingface token
in your Huggingface account settings page.
