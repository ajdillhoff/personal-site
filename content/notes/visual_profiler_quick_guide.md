+++
title = "NVIDIA Visual Profiler Quickstart Guide"
authors = ["Alex Dillhoff"]
date = 2024-04-15T20:14:00-05:00
tags = ["gpgpu"]
draft = false
+++

# NVIDIA Visual Profiler Quickstart Guide

NVIDIA Visual Profiler is installed on both the GPU machines and the workstations. The following guide will show you how to use the NVIDIA Visual Profiler to profile your CUDA code. For more details, please refer to the [official documentation](https://docs.nvidia.com/cuda/profiler-users-guide/index.html).

## Generate a profiling report for your kernel

First, a profiling report must be generated on the machine with a GPU. The following code should replace the current code in `benchmark.sh` for Lab 3. Note that this is specific to the lab in ERB 125. If you're running on your own machine with a GPU more recent than Pascal, it is highly recommended that you profile your code with [NSight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) instead.

The given script will first load CUDA Toolkit 11.5, which is compatible with the version installed on the workstations. It will then compile the code and run the benchmark with a 1024x1024x1024 matrix. The `nvprof` command will generate a profiling report in the form of a `.nvvp` file, which can be opened with NVIDIA Visual Profiler.

```bash
#!/bin/bash
#SBATCH --export=/usr/local/cuda-11.5/bin
#SBATCH --gres=gpu:1

module load cuda/11.5

make benchmark

nvprof --analysis-metrics --export-profile matmul_benchmark.nvvp -f ./build/main/benchmark 1024 1024 1024

module unload cuda/11.5
```

## Open the profiling report

After running the script, you should have a file called `matmul_benchmark.nvvp`. Copy this file from the GPU machine to your local workstation first. You can open this file with the following command:

```bash
nvvp matmul_benchmark.nvvp
```

We will review the metrics reported from the report in class. You can use the provided guided analysis to get a feel for the output metrics and how to interpret them.