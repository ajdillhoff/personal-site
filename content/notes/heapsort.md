+++
title = "Heapsort"
authors = ["Alex Dillhoff"]
date = 2024-02-21T14:58:00-06:00
tags = ["algorithms"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Maintaining the Heap Property](#maintaining-the-heap-property)

</div>
<!--endtoc-->

-   Running time is \\(O(n \lg n)\\).
-   Sorts in place, only a constant number of elements needed in addition to the input.
-   Manages data with a **heap**.

A **binary heap** can be represented as a binary tree, but is stored as an array. The root is the first element of the array. The left subnode for the element at index \\(i\\) is located at \\(2i\\) and the right subnode is located at \\(2i + 1\\). **This assumes a 1-based indexing**.

Using 0-based indexing, we can use \\(2i + 1\\) for the left and \\(2i + 2\\) for the right. The parent could be accessed via \\(\lfloor \frac{i-1}{2} \rfloor\\).

{{< figure src="/ox-hugo/2024-02-21_15-22-37_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>A binary tree as a heap with its array representation (Cormen et al. 2022)." >}}

Heaps come in two flavors: **max-heaps** and **min-heaps**. They can be identified by satisfying a **heap property**.

1.  **max-heap property**: \\(A[parent(i)] \geq A[i]\\)
2.  **min-heap property**: \\(A[parent(i)] \leq A[i]\\)

These properties imply that the root is the largest element in a max-heap and the smallest element in a min-heap.

When it comes to heapsort, a max-heap is used. Min-heaps are used in priority queues. These notes will cover both.


## Maintaining the Heap Property {#maintaining-the-heap-property}

When using heapsort, the heap should always satisfy the **max-heap** property. This relies on a procedure called `max_heapify`. This function assumes that the root element may violate the max-heap property, but the subtrees rooted by its subnodes are valid max-heaps. The function then swaps nodes down the tree until the misplaced element is in the correct position.

```python
def max_heapify(A, i, heap_size):
    l = left(i)
    r = right(i)
    largest = i
    if l < heap_size and A[l] > A[i]:
        largest = l
    if r < heap_size and A[r] > A[largest]:
        largest = r
    if largest != i:
        A[i], A[largest] = A[largest], A[i]
        max_heapify(A, largest, heap_size)
```


### Analysis of Max Heapify {#analysis-of-max-heapify}

Given that `max_heapify` is a recursive function, we can analyze it with a recurrence. The driving function in this case would be the fix up that happens between the current node and its two subnodes, which is a constant time operation. The recurrence is based on how many elements are in the subheap rooted at the current node.

In the worst case of a binary tree, the last level of the tree is half full. That means that the left subtree has height \\(h + 1\\) compared to the right subtree's height of \\(h\\). For a tree of size \\(n\\), the left subtree has \\(2^{h+2}-1\\) nodes and the right subtree has \\(2^{h+1}-1\\) nodes. This is based on a geometric series.

We now have that the number of nodes in the tree is equal to \\(1 + (2^{h+2}-1) + (2^{h+1}-1)\\).

\begin{align\*}
n &= 1 + 2^{h+2} - 1 + 2^{h+1} - 1 \\\\
n &= 2^{h+2} + 2^{h+1} - 1 \\\\
n &= 2^{h+1}(2 + 1) - 1 \\\\
n &= 3 \cdot 2^{h+1} - 1
\end{align\*}

This implies that \\(2^{h+1} = \frac{n+1}{3}\\). That means that, in the worst case, the left subtree would have \\(2^{h+2} - 1 = \frac{2(n+1)}{3} - 1\\) nodes which is bounded by \\(\frac{2n}{3}\\). Thus, the recurrence for the worst case of `max_heapify` is \\(T(n) = T(\frac{2n}{3}) + O(1)\\).
