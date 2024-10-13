+++
title = "Introduction to Algorithms"
authors = ["Alex Dillhoff"]
date = 2023-09-19T00:00:00-05:00
tags = ["computer science", "algorithms"]
draft = false
lastmod = 2024-10-12
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction to Algorithms](#introduction-to-algorithms)
- [Insertion Sort](#insertion-sort)
- [Example: Sorting Numbers](#example-sorting-numbers)
- [Correctness](#correctness)
- [Worst-Case Analysis](#worst-case-analysis)
- [Best-Case Analysis](#best-case-analysis)
- [Rate of Growth](#rate-of-growth)
- [Example: Analysis of Selection Sort](#example-analysis-of-selection-sort)

</div>
<!--endtoc-->



## Introduction to Algorithms {#introduction-to-algorithms}

One of the major goals of computer science is to solve important problems. In order to do that, we must be able to express those solutions both mathematically and in a way that can be executed by a computer. Further, those solutions need to be aware of the resources that are available to them. It does us no good to come up with a solution that could never be run by current hardware or executed in a reasonable amount of time.

There are of course other considerations besides runtime. How much memory does the solution require? Does it require a lot of data to be stored on disk? What about distributed solutions that can be run on multiple machines? Some solutions can be so complex, that we must also consider their environmental impact. For example, Meta's Llama 2 large language models required 3,311,616 combined GPU hours to train. They report that their total carbon emissions from training were 539 tons of CO2 equivalent (<a href="#citeproc_bib_item_1">Touvron et al. 2023</a>).


### What is an algorithm? {#what-is-an-algorithm}

Generally speaking, an **algorithm** is a sequence of steps describing some computational process. It is more abstract that a single function, and an algorithm will typically call many functions to accomplish its task. Algorithms give us a way of translating complex processes from human language into a form that can be executed by a computer.

We begin our algorithmic journey by studying a simple sorting algorithm, insertion sort. First, we need to formally define the problem of sorting. Given a sequence of \\(n\\) objects \\(A = \langle a\_1, a\_2, \ldots, a\_n \rangle\\), we want to rearrange the elements such that \\(a\_1' \leq a\_2' \leq \ldots \leq a\_n'\\). We will assume that the elements are comparable, meaning that we can use the operators \\(<\\) and \\(>\\) to compare them. Some sets, such as the set of all real numbers, have a natural ordering. A useful programming language would provide the required comparison operators. For other types of elements, such as strings, this may not be the case. For example, how would you compare the strings "apple" and "banana"? In these cases, we will need to define our own comparison operators. Either way, we will assume that the comparison operators are available to us.

This example follows the one given in Chapter 2 of Cormen et al. (2009).


## Insertion Sort {#insertion-sort}

Insertion sort is defined as

```python
def insertion_sort(A):
    for i in range(1, len(A)):
        key = A[i]
        j = i - 1
        while j >= 0 and A[j] > key:
            A[j + 1] = A[j]
            j = j - 1
        A[j + 1] = key
```


## Example: Sorting Numbers {#example-sorting-numbers}

Using the algorithm above, sort the following set of numbers. Show the output of the set each time an element is swapped.

\\[
S = \\{10, 4, 7, 0, 2\\}
\\]

\begin{align\*}
\\{10, 4, 7, 0, 2\\}\\\\
\\{4, 10, 7, 0, 2\\}\\\\
\\{4, 7, 10, 0, 2\\}\\\\
\\{4, 7, 0, 10, 2\\}\\\\
\\{4, 0, 7, 10, 2\\}\\\\
\\{0, 4, 7, 10, 2\\}\\\\
\\{0, 4, 7, 2, 10\\}\\\\
\\{0, 4, 2, 7, 10\\}\\\\
\\{0, 2, 4, 7, 10\\}\\\\
\end{align\*}


## Correctness {#correctness}

When coming up with novel algorithms, we may find that it works with a specific use-case, but not in general. It is important to verify that the algorithm works for all possible inputs. Proving the **correctness** of an algorithm may be tricky in some cases, but we can utilize a few techniques to make it easier. The first such technique is called a **loop invariant**. A loop invariant is a statement that is true before and after each iteration of a loop.

The loop invariant for insertion sort is that the subarray \\(A[0 : i - 1]\\) is sorted. This is chosen based on the task. In some cases, the loop invariant is clear from the problem statement. In others, it may require some thought. It is also possible that more than one loop invariant exists for a given algorithm. To determine if a loop invariant is correct, we must verify three things:

1.  **Initialization**: The loop invariant is true before the first iteration of the loop.
2.  **Maintenance**: If the loop invariant is true before an iteration of the loop, it remains true after the iteration.
3.  **Termination**: When the loop terminates, the loop invariant is true.

As long as we can show that the properties above hold, we can be confident that the algorithm is correct. Let's verify the loop invariant for insertion sort.


### Initialization {#initialization}

Before the first iteration of the loop, \\(i = 1\\). The subarray \\(A[0 : i - 1] = A[0 : 0] = \langle A[0] \rangle\\). Since a single element is always sorted, the loop invariant is true before the first iteration.


### Maintenance {#maintenance}

Assume that the loop invariant is true before the $i$th iteration of the loop. That is, \\(A[0 : i - 1]\\) is sorted. We need to show that the loop invariant remains true after the $i + 1$th iteration. The $i + 1$th iteration of the loop will swap elements in the subarray \\(A[0 : i]\\). The loop invariant is maintained if the subarray \\(A[0 : i]\\) is sorted after the $i + 1$th iteration. This is true because the $i + 1$th iteration will swap elements in the subarray \\(A[0 : i]\\) until the element at index \\(i\\) is in the correct position. Therefore, the loop invariant is maintained.


### Termination {#termination}

The loop terminates when \\(i = n\\). At this point, the subarray \\(A[0 : n - 1]\\) is sorted. Since the loop invariant is true after the last iteration, the algorithm is correct.


## Worst-Case Analysis {#worst-case-analysis}

Given the definition from above, we can compute \\(T(n)\\), the running time of the algorithm on an input of size \\(n\\). To do this, we need to sum the products of the cost of each statement and the number of times each statement is executed.

At first glance, the first statement `for i in range(1, len(A))` appears to execute \\(n-1\\) times since it starts at 1 and only goes up to, but not including, \\(n\\). Remember that the `for` statement must be checked to see if it should exit, so the test is executed one more time than the number of iterations. Therefore, the first statement is executed \\(n\\) times. If we say that the cost to execute each check is \\(c\_1\\), then the total cost of the first statement is \\(c\_1 n\\).

With the exception of the `while` loop, the statement inside the `for` loop is executed once per iteration. The cost of executing statement \\(i\\) is \\(c\_i\\). Therefore, the total cost of the second statement is \\(c\_2 n\\). The costs are updated in the code below.

```python
def insertion_sort(A):
    for i in range(1, len(A)): # c_1 n
        key = A[i] # c_2 n
        j = i - 1 # c_3 n
        while j >= 0 and A[j] > key:
            A[j + 1] = A[j]
            j = j - 1
        A[j + 1] = key # c_7 n
```

For the `while` loop, we can denote the number of times it runs by \\(t\_i\\), where \\(i\\) is the iteration of the `for` loop. If the `while` condition check costs \\(c\_4\\) and is executed \\(t\_i\\) times for each `for` loop iteration, the total cost is given as \\(c\_4 \sum\_{i=1}^{n-1} t\_i\\).

The statements inside the `while` loop are executed 1 fewer times than the number of times the condition check is executed. Therefore, the total cost of the statements inside the `while` loop is \\(c\_5 \sum\_{i=1}^{n-1} (t\_i - 1) + c\_5 \sum\_{i=1}^{n-1} (t\_i - 1)\\). The cost of the `while` loop is updated in the code below.

```python
def insertion_sort(A):
    for i in range(1, len(A)): # c_1 * n
        key = A[i] # c_2 * (n-1)
        j = i - 1 # c_3 * (n-1)
        while j >= 0 and A[j] > key: # c_4 * [t_i for i in range(1, len(A))]
            A[j + 1] = A[j] # c_5 * [t_i - 1 for i in range(1, len(A))]
            j = j - 1 # c_6 * [t_i - 1 for i in range(1, len(A))]
        A[j + 1] = key # c_7 * (n-1)
```

To get the total running time \\(T(n)\\), we sum up all of the costs.

\begin{align}
T(n) &= c\_1 n + c\_2 (n-1) + c\_3 (n-1) + c\_4 \sum\_{i=1}^{n-1} t\_i + c\_5 \sum\_{i=1}^{n-1} (t\_i - 1) + c\_6 \sum\_{i=1}^{n-1} (t\_i - 1) + c\_7 (n-1) \\\\
\end{align}

This analysis is a good start, but it doesn't paint the whole picture. The number of actual executions will depend on the input that is given. For example, what if the input is already sorted, or given in reverse order? It is common to express the worst-case runtime for a particular algorithm. For insertion sort, that is when the input is in reverse order. In this case, each element \\(A[i]\\) is compared to every other element in the sorted subarray. This means that \\(t\_i = i\\) for every iteration of the `for` loop. Therefore, the worst-case runtime is given as

\begin{align}
T(n) &= c\_1 n + c\_2 (n-1) + c\_3 (n-1) + c\_4 \sum\_{i=1}^{n-1} i + c\_5 \sum\_{i=1}^{n-1} (i - 1) + c\_6 \sum\_{i=1}^{n-1} (i - 1) + c\_7 (n-1) \\\\
\end{align}

To express this runtime solely in terms of \\(n\\), we can use the fact that \\(\sum\_{i=1}^{n-1} i = (\sum\_{i=0}^{n-1} i) - 1 =  \frac{n(n-1)}{2} - 1\\) and \\(\sum\_{i=1}^{n-1} (i - 1) = \sum\_{i=0}^{n-2} i = \frac{n(n-1)}{2}\\). This gives us

\begin{align}
T(n) &= c\_1 n + c\_2 (n-1) + c\_3 (n-1) + c\_4 \left(\frac{n(n-1)}{2} - 1\right)\\\\
     &+ c\_5 \left(\frac{n(n-1)}{2}\right) + c\_6 \left(\frac{n(n-1)}{2}\right) + c\_7 (n-1) \\\\
        &= \left(\frac{c\_4}{2} + \frac{c\_5}{2} + \frac{c\_6}{2}\right)n^2 + \left(c\_1 + c\_2 + c\_3 + \frac{c\_4}{2} - \frac{c\_5}{2} - \frac{c\_6}{2} + c\_7\right)n - (c\_2 + c\_3 + c\_4 + c\_7) \\\\
\end{align}

With the appropriate choice of constants, we can express this as a quadratic function \\(an^2 + bn + c\\).


## Best-Case Analysis {#best-case-analysis}

The best-case runtime for insertion sort is when the input is already sorted. In this case, the `while` check is executed only once per iteration of the `for` loop. That is, \\(t\_i = 1\\) for every iteration of the `for` loop. Therefore, the best-case runtime is given as

\begin{align}
T(n) &= c\_1 n + c\_2 (n-1) + c\_3 (n-1) + c\_4 (n-1) + c\_7 (n-1) \\\\
     &= (c\_1 + c\_2 + c\_3 + c\_4 + c\_7)n - (c\_2 + c\_3 + c\_4 + c\_7) \\\\
\end{align}

Let \\(a = c\_1 + c\_2 + c\_3 + c\_4 + c\_7\\) and $b = -(c_2 + c_3 + c_4 + c_7)$Then the best-case runtime is given as \\(an + b\\), a linear function of \\(n\\).


## Rate of Growth {#rate-of-growth}

We can simplify how we express the runtime of both these cases by considering only the highest-order term. Consider the worst-case, \\(T(n) = an^2 + bn + c\\). As \\(n\\) grows, the term \\(an^2\\) will dominate the runtime, rendering the others insignificant by comparison. This simplification is typically expressed using \\(\Theta\\) notation. For the worst-case, we say that \\(T(n) = \Theta(n^2)\\). It is a compact way of stating that the runtime is proportional to \\(n^2\\) for large values of \\(n\\).


## Example: Analysis of Selection Sort {#example-analysis-of-selection-sort}

Based on the analysis above, let's check our understanding and see if we can characterize the runtime of another sorting algorithm, selection sort. Selection sort is defined as

```python
def selection_sort(A):
    for i in range(0, len(A) - 1):
        min_index = i
        for j in range(i + 1, len(A)):
            if A[j] < A[min_index]:
                min_index = j
        A[i], A[min_index] = A[min_index], A[i]
```

The first statement `for i in range(0, len(A) - 1)` will be evaluated \\(n\\) times. With the exception of the inner `for` loop, the rest of the statements in the scope of the first `for` loop are executed once per iteration. Their costs are \\(c\_2\\) and \\(c\_6\\), respectively.

The inner `for` loop will be checked \\(n-i\\) times for each iteration of the outer `for` loop. The cost of the condition check is \\(c\_3\\). The cost of the statements inside the `for` loop are \\(c\_4\\) and \\(c\_5\\). The `if` check is evaluated for every iteration of the inner loop, but the statements inside the `if` are only executed when the condition is true. We can denote this as \\(t\_i\\), the number of times the `if` condition is true for each iteration of the inner `for` loop. The cost of the inner loop is given as

\begin{align}
c\_3 \sum\_{i=1}^{n-1} (n-i) + c\_4 \sum\_{i=0}^{n-1} (n-i-1) + c\_5 \sum\_{i=0}^{n-1} t\_i\\\\
\end{align}

Combining this with the cost of the outer `for` loop, we get

\begin{align}
T(n) &= c\_1 n + c\_2 (n-1) + c\_6 (n-1) + c\_3 \sum\_{i=0}^{n-1} (n-i) + c\_4 \sum\_{i=0}^{n-1} (n-i-1) + c\_5 \sum\_{i=0}^{n-1} t\_i\\\\
\end{align}

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Touvron, Hugo, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, et al. 2023. “Llama 2: Open Foundation and Fine-Tuned Chat Models.” arXiv. <a href="https://doi.org/10.48550/arXiv.2307.09288">https://doi.org/10.48550/arXiv.2307.09288</a>.</div>
</div>
