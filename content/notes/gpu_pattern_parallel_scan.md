+++
title = "GPU Pattern: Parallel Scan"
authors = ["Alex Dillhoff"]
date = 2024-02-14T20:09:00-06:00
tags = ["gpgpu"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [What is it?](#what-is-it)
- [Naive Parallel Reduction](#naive-parallel-reduction)
- [Kogge-Stone Algorithm](#kogge-stone-algorithm)

</div>
<!--endtoc-->



## What is it? {#what-is-it}

-   Parallelizes sequential problems.
-   Works with computations that can be described in terms of a recursion.
-   Used as a primitive operation for sorting, tree operations, and recurrences.
-   Studying this will also reveal how parallelization can increase the complexity beyond that of a traditional sequential approach.


### Example: Inclusive Scan {#example-inclusive-scan}

Given an array of numbers, the inclusive scan computes the sum of all elements up to a given index. For example, given the array [1, 2, 3, 4, 5], the inclusive scan would produce [1, 3, 6, 10, 15]. You could solve this recursively, but it would be horribly inefficient. A sequential solution is achievable with dynamic programming. However, a parallel solution is much more efficient.

```cpp
void sequential_scan(float *x, float *y, uint N) {
    y[0] = x[0];
    for (uint i = 1; i < N; i++) {
        y[i] = y[i - 1] + x[i];
    }
}
```


## Naive Parallel Reduction {#naive-parallel-reduction}

If we have \\(n\\) elements, we could have \\(n\\) threads each compute the sum of a single element. How many operations would that take? The first thread computes the sum of 1 element, or 0 operations. The second thread computes the sum of 2 elements, 1 operation, and so on. This can be described as a sum of the first \\(n\\) natural numbers, which is \\(n(n + 1)/2\\). This parallel solution is worse than the sequential solution, coming in at \\(O(n^2)\\).


## Kogge-Stone Algorithm {#kogge-stone-algorithm}

The first solution to this problem relies on [GPU Pattern: Reduction]({{< relref "gpu_pattern_reduction.md" >}}) and is called the Kogge-Stone algorithm. The algorithm was published in 1973 by Peter M. Kogge and Harold S. Stone during their time at Stanford University.


### Adapting the Reduction Tree {#adapting-the-reduction-tree}

Design reduction tree so that each thread has access to relevant inputs. The input matrix is modified to that input \\(A\_i\\) contains the sum of up to \\(2^k\\) elements after \\(k\\) iterations. For example, after iteration 2, \\(A\_3\\) contains the sum \\(A\_0 + A\_1 + A\_2 + A\_3\\).

{{< figure src="/ox-hugo/2024-02-18_18-31-36_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Visualization of parallel inclusive scan based on the Kogge-Stone algorithm (Source: NVIDIA DLI)." >}}

This is implemented in the following code:

```cuda
__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N) {
        __shared__ float A[SECTION_SIZE];
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N) {
            A[threadIdx.x] = X[i];
        } else {
            A[threadIdx.x] = 0;
        }

        for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
            __syncthreads();
            float temp;
            if (threadIdx.x >= stride) {
                temp = A[threadIdx.x] + A[threadIdx.x - stride];
            }
            __syncthreads();
            if (threadIdx.x >= stride) {
                A[threadIdx.x] = temp;
            }
        }

        if (i < N) {
            Y[i] = A[threadIdx.x];
        }
}
```

It is not possible to guarantee that a block can cover the entire input, so `SECTION_SIZE` is used to ensure that the input is covered. This should be the same as the block size. Each thread starts off by loading its initial input into shared memory. Starting with thread 2, each thread computes a sum of the value assigned to its thread as well as the one before it by a factor of `stride`.

The loop itself is moving _down_ the reduction tree, which is bounded logarithmically. The local variable `temp` is used to store the intermediate result before barrier synchronization takes place. Otherwise there would be a possibility of a _write-after-read_ race condition.


### Double Buffering {#double-buffering}

The temporary variable and second call to `__syncthreads()` are necessary since a thread may read from a location that another thread is writing to. If the input and output arrays were represented by two different areas of shared memory, this call could be removed. This approach is called **double-buffering**.

It works as follows: the input array is read from global memory into shared memory. At each iteration the data read from the first array is used to write new values to the second array. Since the values used in that iteration are only read from the first array, the second array can be used as the input array for the next iteration. This cycle continues back and forth until the final result is written to the output array.
