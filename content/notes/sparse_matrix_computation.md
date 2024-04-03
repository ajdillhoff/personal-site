+++
title = "Sparse Matrix Computation"
authors = ["Alex Dillhoff"]
date = 2024-03-30T10:36:00-05:00
tags = ["gpgpu"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Coordinate List Format (COO)](#coordinate-list-format--coo)
- [Compressed Sparse Row Format (CSR)](#compressed-sparse-row-format--csr)
- [ELL Format](#ell-format)
- [ELL-COO Format](#ell-coo-format)
- [Jagged Diagonal Storage Format (JDS)](#jagged-diagonal-storage-format--jds)

</div>
<!--endtoc-->



## Introduction {#introduction}

Sparse matrices are matrices with mostly zero elements. They are common in scientific computing, machine learning, and other fields. It is important to study them in the context of GPU computing because they can be very large and require a lot of memory. Effeciently representing and computing with sparse matrices provides a substantial benefit to many applications.

The obvious benefits of sparsity is that we can typically represent the same matrix using a smaller memory footprint. Fewer elements means fewer wasted operations as well. The challenges of GPU implementations are that the memory access patterns are not always friendly to the GPU architecture. This is especially true for sparse matrices, where the memory access patterns are often irregular.

These notes will review the sparse matrix formats as presented in (<a href="#citeproc_bib_item_1">Hwu, Kirk, and El Hajj 2022</a>). Each will be evaluated using the following criteria:

1.  **Compaction**: How well does the format compact the data?
2.  **Flexibility**: Is the format easy to modify?
3.  **Accessibility:** How easy is it to access the data?
4.  **Memory access efficiency:** Are the accesses coalesced?
5.  **Load balance:** Are the operations balanced across threads?


## Coordinate List Format (COO) {#coordinate-list-format--coo}

This format stores non-zero elements in a 1D array of values. It also requires two 1D arrays to store the row and column indices, incurrent an overhead of 2N. The values in each array are contiguous, which is good for memory access.

{{< figure src="/ox-hugo/2024-04-02_19-42-49_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>COO Format (<a href=\"#citeproc_bib_item_1\">Hwu, Kirk, and El Hajj 2022</a>)." >}}


### Kernel Implementation {#kernel-implementation}

```cuda
__global__ void spmv_coo_kernel(COOMatrix cooMatrix, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cooMatrix.numNonzeros) {
        int row = cooMatrix.rowIdx[i];
        int col = cooMatrix.colIdx[i];
        float val = cooMatrix.values[i];
        atomicAdd(&y[row], val * x[col]);
    }
}
```


### Evaluation {#evaluation}

1.  **Compaction**: Compared to representing the matrices in dense format, the COO format is very compact. However, it is not as compact as some other sparse matrix formats. It requires an additional over head of 2N elements to store the row and column indices.
2.  **Flexibility:** Indices and values can be easily modified in this format. This is good for applications that require frequent modifications.
3.  **Accessibility:** It is easy to access nonzero elements. It is **not** easy to access the original 0s in each row.
4.  **Memory access efficiency:** The values in this format are contiguous, resulting in coalesced memory access.
5.  **Load balance:** The data is uniformly distributed across threads, resulting in good load balance.

One major drawback, as seen in the code above, is the use of atomic operations.


## Compressed Sparse Row Format (CSR) {#compressed-sparse-row-format--csr}

The key idea of this format is that each thread is responsible for all nonzeros in a row.

{{< figure src="/ox-hugo/2024-04-02_19-50-47_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>CSR Format (<a href=\"#citeproc_bib_item_1\">Hwu, Kirk, and El Hajj 2022</a>)." >}}


### Kernel Implementation {#kernel-implementation}

```cuda
__global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < csrMatrix.numRows) {
        float sum = 0.0f;
        for (int j = csrMatrix.rowPtr[row]; j < csrMatrix.rowPtr[row + 1]; j++) {
            sum += csrMatrix.values[j] * x[csrMatrix.colIdx[j]];
        }
        y[i] = sum;
    }
}
```

The rows are mapped to a single pointer index, so it only needs \\(m\\) entries to store them. The columns are not required to be in order. If the columns _are_ in order, the data is represented in row-major order without the zero elements.


### Evaluation {#evaluation}

1.  **Compaction**: The CSR format is more compact than the COO format since it only requires \\(m\\) entries to store the row pointers.
2.  **Flexibility:** The CSR format is not as flexible as the COO format. It is not easy to modify the values or indices.
3.  **Accessibility:** There is less parallelization than COO due to the row sizes.
4.  **Memory access efficiency:** The memory access pattern is poor since the data is separated over columns.
5.  **Load balance:** The load is not balanced across threads. Some threads will have more work than others, leading to control divergence.


## ELL Format {#ell-format}

ELL fixes the non-coalesced memory accesses of CSR via data padding and transposition. This is visualized below:

1.  Start with CSR format
2.  Pad rows to equal size
3.  Store in column-major order

{{< figure src="/ox-hugo/2024-04-02_19-53-05_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>ELL Format (<a href=\"#citeproc_bib_item_1\">Hwu, Kirk, and El Hajj 2022</a>)." >}}


### Kernel Implementation {#kernel-implementation}

```cuda
__global__ void spmv_ell_kernel(ELLMatrix ellMatrix, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < ellMatrix.numRows) {
        float sum = 0.0f;
        for (int j = 0; j < ellMatrix.nnzPerRow[row]; j++) {
            int col = ellMatrix.colIdx[j * ellMatrix.numRows + row];
            sum += ellMatrix.values[j * ellMatrix.numRows + row] * x[col];
        }
        y[row] = sum;
    }
}
```


### Evaluation {#evaluation}

1.  **Compaction:** Padding the rows means this is less space efficient than CSR.
2.  **Flexibility:** More flexible than CSR; adding nonzeros in CSR requires a shift of values. This format can replaced a padded element if necessary.
3.  **Accessibility:** ELL can return the row given the index of a nonzero element as well as the nonzero of a row given that index.
4.  **Memory access efficiency:** Consecutive threads access consecutive memory locations.
5.  **Load balance:** Shares the same control divergence issues as CSR.


## ELL-COO Format {#ell-coo-format}

ELL-COO combines the two formats to improve space efficiency and control divergence.

{{< figure src="/ox-hugo/2024-04-02_20-01-43_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>ELL-COO Format (<a href=\"#citeproc_bib_item_1\">Hwu, Kirk, and El Hajj 2022</a>)." >}}


### Evaluation {#evaluation}

1.  **Compaction:** ELL-COO has the same compaction as ELL.
2.  **Flexibility:** ELL-COO is more flexible than ELL thanks to inclusion of the COO format.
3.  **Accessibility:** It is not always possible to access all nonzeros given a row index.
4.  **Memory access efficiency:** The memory access pattern is coalesced.
5.  **Load balance:** COO reduces the control divergence seen in ELL alone.


## Jagged Diagonal Storage Format (JDS) {#jagged-diagonal-storage-format--jds}

The last format we will consider is the Jagged Diagonal Storage format. This format reduces divergence and improves memory coalescing without padding. The main idea is to sort the rows by length from longest to shortest.

1.  Group nonzeros by row
2.  Sort rows by length while preserving their original row indices
3.  Store in column-major order

{{< figure src="/ox-hugo/2024-04-02_20-07-30_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>JDS Format (<a href=\"#citeproc_bib_item_1\">Hwu, Kirk, and El Hajj 2022</a>)." >}}


### Evaluation {#evaluation}

1.  **Compaction:** Avoid paddding, so it is more space efficient than ELL.
2.  **Flexibility:** Less flexible than ELL since it requires sorting when adding new elements.
3.  **Accessibility:** Cannot access a row and column given the index of a nonzero element.
4.  **Memory access efficiency:** Without padding, the starting location of memory accesses in each iteration can vary.
5.  **Load balance:** Since the rows are sorted, threads of the same warp are likely to iterate over rows of similar length.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Hwu, Wen-mei W., David B. Kirk, and Izzat El Hajj. 2022. <i>Programming Massively Parallel Processors: A Hands-on Approach</i>. Fourth. Morgan Kaufmann.</div>
</div>
