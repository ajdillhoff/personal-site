+++
title = "Parallel Sorting Algorithms"
authors = ["Alex Dillhoff"]
date = 2024-03-31T10:50:00-05:00
tags = ["gpgpu"]
draft = false
sections = "GPU Programming"
lastmod = 2025-04-06
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Radix Sort](#radix-sort)
- [Optimizing Memory Access Efficiency](#optimizing-memory-access-efficiency)
- [Choosing a different Radix value](#choosing-a-different-radix-value)

</div>
<!--endtoc-->



## Radix Sort {#radix-sort}

For a background on Radix Sort, see these notes on [Sorting in Linear Time]({{< relref "sorting_in_linear_time.md" >}}).

Radix sort relies on counting sort for each section, and each section must be processed before moving onto the next. The parallel solution will not attempt to address this sequential dependency. Instead, we will focus on the parallelization of the counting sort step.

Each thread must determine where to place its input elements. For each bit, the thread will assign it to either a 0 or 1 bucket. Since all values will either be 0 or 1, the thread needs to compute the number of 0s and 1s that come before it in the current section. Radix sort is also a stable sort, so the order of elements with the same key must be preserved. Consider the following array separated into 4 threads of 4 elements each:

\begin{array}{l|cccc|cccc|cccc|cccc}
Value & 0 & 1 & 0 & 1 & 1 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 1 & 0 & 1 & 0\\\\
Index & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15\\\\
\end{array}

The least significant bits for each thread are \\([1, 0, 1, 0]\\). From thread 4's perspective, there are 2 1s and a single 0 that come before it. Its key index is 3 (using 0-based indexing), so it only needs to compute the number of 1s that come before it and subtract that from its key index: \\(3 - 2 = 1\\). More generally, for a 0 bit:

\\[
\text{output index} = \text{key index} - \text{number of 0s that come before it}
\\]

The calculation for the 1 bit hinges on the fact that all keys mapping to 0 must come before it.

\\[
\text{output index} = \text{input size} - \text{number of ones total} + \text{number of 1s that come before it}
\\]

```c
__global__ void radix_sort_iter(unsigned int *input, unsigned int *output,
                                unsigned int *bits, unsigned int N, unsigned int iter) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int key, bit;

    if (idx < N) {
        key = input[i];
        bit = (key >> iter) & 1;
        bits[i] = bit;
    }

    exclusiveScan(bits, N);

    if (idx < N) {
        unsigned int numOnesBefore = bits[idx];
        unsigned int numOnesTotal = bits[N];
        unsigned int dst = (bit == 0) ? idx - numOnesBefore
                                      : N - numOnesTotal + numOnesBefore;
        output[dst] = key;
    }
}
```


### Example {#example}

Consider the fourth thread with `idx = 3` using the array from above. This thread index is certainly less than the array size, so the `key` is read from `input` before extracting the least significant bit. **Note that there is a call to thread synchronization inside `exclusiveScan`.** The result of `exclusiveScan` is an array that indicates, for each index, the number of ones that came before it. For our array, this is:

\\[
[0, 1, 1, 2]
\\]

The destination can be computed for each thread. The result is shown in the table below.

| Thread       | 0 | 1 | 2 | 3 |
|--------------|---|---|---|---|
| Bit          | 1 | 0 | 1 | 0 |
| #1s Before   | 0 | 1 | 1 | 2 |
| #1s Total    | 2 | 2 | 2 | 2 |
| Output Index | 2 | 0 | 3 | 1 |


## Optimizing Memory Access Efficiency {#optimizing-memory-access-efficiency}

Each thread write their keys to global memory in an uncoalesced manner. This can be optimized by having each block maintain local buckets in shared memory. The keys within each block will be coalesced when written to global memory.

TODO: Show visualization similar to 13.5

In order to make this work, each thread needs to calculate where in the output array the values from its bucket should be placed. For 0 bits, the block's 0 bucket will come after the 0 buckets from all previous blocks. These positions can be computed by performing an exclusive scan on the block's local bucket sizes.

TODO: Show visualization similar to 13.6


## Choosing a different Radix value {#choosing-a-different-radix-value}

Picking a larger radix value will reduce the number of iterations required to sort the array. In the previous example with 4-bit keys, we can split them up with a radix value of 2 bits. This will require 2 iterations to sort the array. Instead of 2 buckets for 0 or 1, we will have 4 buckets for 00, 01, 10, and 11. All keys with 00 will come before 01, and so on.
