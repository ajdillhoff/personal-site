+++
title = "Dynamic Parallelism"
authors = ["Alex Dillhoff"]
date = 2024-04-19T16:52:00-05:00
tags = ["gpgpu", "cuda"]
draft = false
+++

**Dynamic Parallelism** is an extension to CUDA that enables kernels to directly call other kernels. Earlier versions of CUDA only allowed kernels to be launched from the host code. When we studied <GPU Pattern: Parallel Scan>, the segmented approach required multiple kernel calls.
