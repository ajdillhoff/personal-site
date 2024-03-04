+++
title = "Quicksort"
authors = ["Alex Dillhoff"]
date = 2024-02-25T17:24:00-06:00
tags = ["algorithms", "computer science"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Basic Quicksort](#basic-quicksort)
- [Performance](#performance)
- [Randomized Quicksort](#randomized-quicksort)
- [Paranoid Quicksort](#paranoid-quicksort)

</div>
<!--endtoc-->

Quicksort is a popular sorting algorithm implemented in many language libraries that has a worst-case running time of \\(\Theta(n^2)\\). **Why would anyone choose this as the default sorting algorithm if one like mergesort has better worst-case performance?** As you will see, the devil is in the details. Quicksort is often faster in practice. It also has a small memory footprint and is easy to implement.

**TODO: Discuss distinct values and the impact on quicksort**


## Basic Quicksort {#basic-quicksort}

Quicksort follows a divide-and-conquer approach to sorting. Given an input array of \\(n\\) elements, it selects a pivot element and partitions the array into two sub-arrays: one with elements less than the pivot and one with elements greater than the pivot. It then recursively sorts the sub-arrays. Since the subarrays are recursively sorted, there is no need for a merge step as in mergesort.

```python
def quicksort(arr, p, r):
    q = partition(arr, p, r)
    quicksort(arr, p, q - 1)
    quicksort(arr, q + 1, r)
```

This algorithm looks deceptively simple. The complexity is hidden in the partitioning step. This procedure will rearrange the elements in the array such that the pivot element is in its final position and all elements less than the pivot are to the left of it and all elements greater than the pivot are to the right of it.


### Partitioning {#partitioning}

For basic quicksort, the first or last element is chosen as the pivot. Picking it this way yields a fairly obvious recurrence of \\(T(n) = T(n-1) + O(n)\\), which is \\(\Theta(n^2)\\). As mentioned before, this algorithm is executed in-place, meaning there is no need for additional memory to store the sub-arrays. It is done through a clever use of indices.

```python
def partition(arr, p, r):
    x = arr[r]
    i = p - 1
    for j in range(p, r):
        if arr[j] <= x:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[r] = arr[r], arr[i + 1]
    return i + 1
```

The indices are used to define the following loop invariant.

1.  **Left:** if \\(p \leq k \leq i\\), then \\(A[k] \leq x\\)
2.  **Middle:** if \\(i + 1 \leq k \leq j - 1\\), then \\(A[k] > x\\)
3.  **Right:** if \\(k = r\\), then \\(A[k] = x\\)

{{< figure src="/ox-hugo/2024-02-25_19-18-19_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Operation of partition (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}

The figure above is from CRLS which shows an example run of the partitioning algorithm. It starts with \\(i, p, j\\) on the left and \\(r\\) on the right. Since the value at \\(j\\) is less than the pivot, \\(i\\) is incremented and the values at \\(i\\) and \\(j\\) are swapped. In the next iteration, the value at \\(j\\) is greater than the pivot, so nothing is done. This continues until \\(j\\) reaches \\(r\\). At this point, the pivot is swapped with the value at \\(i + 1\\).


### Example {#example}

{{< figure src="/ox-hugo/2024-02-26_16-34-22_Quicksort Example.png" caption="<span class=\"figure-number\">Figure 2: </span>Example of Quicksort." >}}

The example above starts with an unsorted array. The second row shows the array after the first call to `partition`. The left and right subarrays are called recursively. The first subarray in row 3 has a pivot 3 and is already partitioned. The right subarray follows the same convenience.

On row 4, the left subarray is modified by swapping the pivot of 2 with 1. The right subarray first swaps 5 and 7 before swapping 6 and 7. The final array is sorted as shown on the last row.


## Performance {#performance}

The performance of quicksort is dependent on the pivot. If the subarrays on either side of the pivot are balanced, the running time is asymptotically similar to mergesort. In the worst case, it will run in quadratic time.

The worst partitioning occurs when the pivot is the smallest or largest element in the array. This creates a subarray of size \\(n - 1\\) and another of size 0. The partitioning itself takes \\(\Theta(n)\\) time, yielding a recurrence of \\(T(n) = T(n - 1) + \Theta(n)\\). We can use the substitution method to solve this recurrence and find that the worst-case running time is \\(\Theta(n^2)\\). This can happen even if the input is already sorted before hand.

Cormen et al. present a recursive analysis of the running time where, at each level, the partition produces a 9-to-1 split. This is visualized in the sketch below.

{{< figure src="/ox-hugo/2024-02-26_18-20-51_Quicksort Recursion 01 Artboard 1 (2).jpg" caption="<span class=\"figure-number\">Figure 3: </span>Recursion tree for quicksort." >}}

The subtree for the \\(\frac{1}{10}\\) split eventually bottoms out after being called \\(\log\_{10} n\\) times. Until this happens, the cost of each level of the tree is \\(n\\). After the left-tree bottoms out, the right tree continues with an upper bound of \\(\leq n\\). The right tree completes after \\(\log\_{10/9} n = \Theta(\lg n)\\) levels. Since each level of the tree cost no more than \\(n\\), the total cost is \\(\Theta(n \lg n)\\).


### Best Case {#best-case}

In the best-case, the pivot is the median of the array and two balanced subarrays are created: one of size \\(n/2\\) and another of size \\(\lfloor (n-1)/2 \rfloor\\). The recurrence is \\(T(n) = 2T(n/2) + \Theta(n)\\), which is \\(\Theta(n \log n)\\).

Using the substitution method, we can show the best-case running time. We start with the fact that the partitioning produces two subproblems with a total size of \\(n-1\\). This gives the following recurrence:

\\[
T(n) = \min\_{0 \leq q \leq n-1} \\{T(q) + T(n - q - 1)\\} + \Theta(n).
\\]

The minimum function accounts for the minimum time taken to sort any partition of the array, where \\(q\\) represents the pivot element's position.

Our **hypothesis** will be that

\\[
T(n) \geq cn \lg n = \Omega(n \lg n).
\\]

Plugging in our hypothesis, we get

\begin{align\*}
T(n) &\geq \min\_{0 \leq q \leq n-1} \\{cq \lg q + c(n - q - 1) \lg (n - q - 1)\\} + \Theta(n) \\\\
& c \min\_{0 \leq q \leq n-1} \\{q \lg q + (n - q - 1) \lg (n - q - 1)\\} + \Theta(n).
\end{align\*}

If we take the derivative of the function inside the minimum with respect to \\(q\\), we get

\\[
\frac{d}{dq} \\{q \lg q + (n - q - 1) \lg (n - q - 1)\\} = c\\{\frac{q}{q} + \lg q - \lg (n - q - 1) - \frac{(n - q - 1)}{(n - q - 1)}\\}.
\\]

Setting this equal to zero and solving for \\(q\\) yields

\\[
q = \frac{n - 1}{2}.
\\]

We can then plug this value of \\(q\\) into the original function to get

\begin{align\*}
T(n) &\geq c \frac{n - 1}{2} \lg \frac{n - 1}{2} + c \frac{n - 1}{2} \lg \frac{n - 1}{2} + \Theta(n) \\\\
&= cn \lg (n - 1) + c (n - 1) + \Theta(n) \\\\
&= cn \lg (n - 1) + \Theta(n) \\\\
&\geq cn \lg n \\\\
&= \Omega(n \lg n).
\end{align\*}


### Average Case {#average-case}

As it turns out, the average-case running time is \\(\Theta(n \log n)\\) (<a href="#citeproc_bib_item_1">Cormen et al. 2022</a>). Quicksort is highly dependent on the relative ordering of the input. Consider the case of a randomly ordered array. The cost of partitioning the original input is \\(O(n)\\). Let's say that the pivot was the last element, yielding a split of 0 and \\(n - 1\\). Now, let's say we get lucky on the next iteration and get a balanced split. Even if the rest of the algorithm splits between the median and the last element, the upper bound on the running time is \\(\Theta(n \log n)\\). It is highly unlikely that the split will be unbalanced on every iteration given a random initial ordering.

We can make this a tad more formal by defining a lucky \\(L(n) = 2U(n/2) + \Theta(n)\\) and an unlucky split \\(U(n) = L(n-1) + \Theta(n)\\). We can solve for \\(L(n)\\) by plugging in the definition of \\(U(n)\\).

\begin{align\*}
L(n) &= 2U(n/2) + \Theta(n) \\\\
&= 2(L(n/2 - 1) + \Theta(n/2)) + \Theta(n) \\\\
&= 2L(n/2 - 1) + \Theta(n) \\\\
&= \Theta(n \log n)
\end{align\*}


## Randomized Quicksort {#randomized-quicksort}

The intuition of the crude analysis above is that we would have to be extremely unlucky to get a quadratic running time if the input is randomly ordered. Randomized quicksort builds on this intuition by selection a random pivot on each iteration. This is done by swapping the pivot with a random element before partitioning.

```python
import random

def randomized_partition(arr, p, r):
    i = random.randint(p, r)
    arr[i], arr[r] = arr[r], arr[i]
    return partition(arr, p, r)

def randomized_quicksort(arr, p, r):
    if p < r:
        q = randomized_partition(arr, p, r)
        randomized_quicksort(arr, p, q - 1)
        randomized_quicksort(arr, q + 1, r)
```


### Analysis {#analysis}

The lightweight analysis above reasoned that, as long as each split puts a constant amount of elements to one side of the split, then the running time is \\(\Theta(n \log n)\\).

We can understand this analysis simply by asking the right questions. First, our primary question: **What is the running time of Quicksort dependent on?** The biggest bottleneck is the partitioning function. At most, we get really unlucky and the first pivot is picked every time. This means it is called \\(n\\) times yielding \\(O(n)\\). The variable part of this is figuring out how many element comparisons are made. The running time is then \\(O(n + X)\\).


#### The Expected Value of \\(X\\) {#the-expected-value-of-x}

The number of comparisons can be expressed as

\\[
X = \sum\_{i=1}^{n-1} \sum\_{j=i+1}^{n} X\_{ij},
\\]

where \\(X\_{ij}\\) is the indicator random variable that is 1 if \\(A[i]\\) and \\(A[j]\\) are compared and 0 otherwise. This works with our worst case analysis. If we always get a split of 0 and \\(n - 1\\), then the indicator random variable is 1 for every comparison, yielding \\(O(n^2)\\). Taking the expectation of both sides:

\begin{align\*}
E[X] &= E\left[\sum\_{i=1}^{n-1} \sum\_{j=i+1}^{n} X\_{ij}\right] \\\\
&= \sum\_{i=1}^{n-1} \sum\_{j=i+1}^{n} E[X\_{ij}] \\\\
&= \sum\_{i=1}^{n-1} \sum\_{j=i+1}^{n} P(X\_{ij} = 1).
\end{align\*}

**What is P(X<sub>ij</sub> = 1)?** Let \\(z\_i, \dots, z\_j\\) be the indices of elements in a sorted version of the array. Under this assumption, \\(z\_i\\) is compared to \\(z\_j\\) only if \\(z\_i\\) or \\(z\_j\\) is the first pivot chosen from the subarray \\(A[i \dots j]\\). In a set of distinct elements, the probability of picking any pivot from the array from \\(i\\) to \\(j\\) is \\(\frac{1}{j - i + 1}\\). This means that the probability of comparing \\(z\_i\\) and \\(z\_j\\) is \\(\frac{2}{j - i + 1}\\). We can now finish the calculation.

\begin{align\*}
E[X] &= \sum\_{i=1}^{n-1} \sum\_{j=i+1}^{n} \frac{2}{j - i + 1} \\\\
&= \sum\_{i=1}^{n-1} \sum\_{k=1}^{n-i} \frac{2}{k + 1} \\\\
&< \sum\_{i=1}^{n-1} \sum\_{k=1}^{n-i} \frac{2}{k} \\\\
&= \sum\_{i=1}^{n-1} O(\log n) \\\\
&= O(n \log n).
\end{align\*}


## Paranoid Quicksort {#paranoid-quicksort}

Repeat the following until the partitioning until the left or right subarray is less than or equal to \\(\frac{3}{4}\\) of the original array.

1.  Choose a random pivot.
2.  Partition the array.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Cormen, Thomas H., Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2022. <i>Introduction to Algorithms</i>. 4th ed. MIT Press. <a href="http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/">http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/</a>.</div>
</div>
