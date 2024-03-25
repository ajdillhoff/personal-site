+++
title = "Greedy Algorithms"
authors = ["Alex Dillhoff"]
date = 2024-03-18T14:45:00-05:00
tags = ["computer science", "algorithms"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Activity Selection](#activity-selection)
- [Properties of Greedy Solutions](#properties-of-greedy-solutions)

</div>
<!--endtoc-->

Greedy algorithms are a class of algorithms that yield _locally_ optimal solutions. In cases where the local optimum is also the global optimum, greedy algorithms are ideal. Even in cases where the global solution is more elusive, a local solution may be sufficient.


## Activity Selection {#activity-selection}

Given a set of activities that need to be scheduled using a common resource, the **activity selection** problem is to find the maximum number of activities that can be scheduled without overlapping.

Each activity has a start time \\(s\_i\\) and finish time \\(f\_i\\), where \\(0 \leq s\_i < f\_i < \infty\\). An activity \\(a\_i\\) takes place over the interval \\([s\_i, f\_i)\\). Two activities \\(a\_i\\) and \\(a\_j\\) are mutually compatible if \\(s\_i \geq f\_j\\) or \\(s\_j \geq f\_i\\).

Sort activities by their finish time. **Objective:** Find the largest subset of mutually compatible activities.

| \\(i\\)    | 1 | 2 | 3 | 4 | 5 | 6  | 7  | 8  | 9  |
|------------|---|---|---|---|---|----|----|----|----|
| \\(s\_i\\) | 1 | 2 | 4 | 1 | 5 | 8  | 9  | 11 | 13 |
| \\(f\_i\\) | 3 | 5 | 7 | 8 | 9 | 10 | 11 | 14 | 16 |

{{< figure src="/ox-hugo/2024-03-22_17-14-45_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Visualization of activities over time (Cormen et al. 2022)." >}}

**How many mutually compatible sets are there?**

1.  \\(\\{a\_1, a\_3, a\_6, a\_8\\}\\)
2.  \\(\\{a\_1, a\_3, a\_6, a\_9\\}\\)
3.  \\(\\{a\_1, a\_3, a\_7, a\_9\\}\\)
4.  \\(\\{a\_1, a\_5, a\_7, a\_8\\}\\)
5.  \\(\\{a\_1, a\_5, a\_7, a\_9\\}\\)
6.  \\(\\{a\_2, a\_5, a\_7, a\_8\\}\\)
7.  \\(\\{a\_2, a\_5, a\_7, a\_8\\}\\)


### Optimal Substructure {#optimal-substructure}

**How do we verify that this problem has optimal substructure?** First, it is important to formalize the problem based on the definition given previously. Define \\(S\_{ij}\\) as the set of all activities that start after \\(a\_i\\) finishes and finish before \\(a\_j\\) starts.

\\[
S\_{ij} = \\{a\_k \in S : f\_i \leq s\_k < f\_k \leq s\_j\\}
\\]

This defines a clear subset of the original set of data. That is, we have defined a subproblem of the original problem.

**Which activities are those in \\(S\_{ij}\\) compatible with?**

1.  any \\(a\_i\\) that finish by \\(f\_i\\)
2.  any \\(a\_i\\) that start no earlier than \\(s\_j\\).

Given this subset, our subproblem is that of finding a maximum set of mutually compatible activities in \\(S\_{ij}\\), denoted \\(A\_{ij}\\). If \\(a\_k \in A\_{ij}\\), we are left with two subproblems:

1.  Find mutually compatible activities in \\(S\_{ik}\\) -- starts after \\(a\_i\\) finishes and finish before \\(a\_k\\) starts.
2.  Find mutually compatible activities in \\(S\_{kj}\\) -- starts after \\(a\_k\\) finishes and finish before \\(a\_j\\) start.

The two subsets above are defined as \\(A\_{ik} = A\_{ij} \cap S\_{ik}\\) and \\(A\_{kj} = A\_{ij} \cap S\_{kj}\\), respectively. Then \\(A\_{ij} = A\_{ik} \cup \\{a\_k\\} \cup A\_{kj}\\). The size of the set is given by \\(|A\_{ij}| = |A\_{ik}| + 1 + |A\_{kj}|\\).

**Claim:** If our problem has optimal substructure, then the optimal solution \\(A\_{ij}\\) must include optimal solutions for \\(S\_{ik}\\) and \\(S\_{kj}\\).

This claim can be proven using the **cut-and-paste** method used in [Dynamic Programming]({{< relref "dynamic_programming.md" >}}). This technique works by showing that if a solution to a problem is not optimal, then there exists a way to **cut** the suboptimal portion and **paste** an optimal one. This will lead to a contradiction because the original was assumed to be optimal.

**Proof:** Suppose that \\(A\_{kj}\\) is not optimal, and we could find a set \\(A'\_{kj}\\) that is larger. Then we could replace \\(A\_{kj}\\) with \\(A'\_{kj}\\) in \\(A\_{ij}\\) to obtain a larger set. This contradicts the assumption that \\(A\_{ij}\\) is optimal.

Simply put, if the claim is that the given solution is optimal, and the solution is constructed from optimal solutions to subproblems, then there cannot exist any other solution that is better. Another way to look at this: if we construct optimal solutions to subproblems, then the solution to the original problem must be optimal.


### Recursive Solution {#recursive-solution}

Let \\(c[i, j]\\) be the size of the optimal solution for \\(S\_{ij}\\). Based on the above discussion, the size is computed as

\\[
c[i, j] = c[i, k] + c[k, j] + 1.
\\]

This dynamic programming solution assumes we know the optimal solution for all subproblems. To know this, we need to examine all possibilities which include \\(a\_k\\) in the solution.

\\[
c[i, j] = \begin{cases}
0 & \text{if } S\_{ij} = \emptyset, \\\\
\max \\{c[i, k] + c[k, j] + 1 : a\_k \in S\_{ij}\\} & \text{if } S\_{ij} \neq \emptyset.
\end{cases}
\\]


### Greedy Solution {#greedy-solution}

The greedy solution is the naive one: select an activity that leaves the resource available for as many other activities as possible, which is the activity that finishes first. If multiple activities finish at the same time, select one arbitrarily.

The subproblem is that of finding a maximum size set of mutually compatible activities that start after \\(a\_1\\) finishes. More generally, the optimal solution consists of all \\(a\_i\\) that start after \\(a\_k\\) finishes, where \\(a\_k\\) is the last activity to finish:

\\[
S\_{k} = \\{a\_i \in S : s\_i \geq f\_k\\}.
\\]

**Is the greedy solution optimal?** Suppose that \\(a\_m \in S\_k\\) is the activity that finishes first. Then it must be included in the maximum size subset of mutually compatible activities \\(A\_k\\). Suppose we are given \\(A\_k\\) and we look at \\(a\_j \in A\_k\\), the activity that finishes first. If \\(a\_j = a\_m\\), then the greedy solution is optimal. If \\(a\_j \neq a\_m\\), then we can replace \\(a\_j\\) with \\(a\_m\\) since they are both compatible with all other activities in \\(A\_k\\). Picking the activity that finishes first does not change the size of the optimal solution.

The solution is top-down:

1.  pick the solution that finishes first,
2.  remove all activities that are incompatible with the chosen activity,
3.  repeat until no activities remain.


### Recursive Greedy Algorithm {#recursive-greedy-algorithm}

```python
def recursive_activity_selector(s, f, k, n):
    m = k + 1
    while m <= n and s[m] < f[k]:
        m += 1
    if m <= n:
        return [m] + recursive_activity_selector(s, f, m, n)
    else:
        return []
```

This algorithm assumes that `f` is sorted in increasing order. The index `k` represents the index of the current subproblem. The number of activities is given by `n`. The `while` loop increments `m` until it finds an activity that starts after activity `k` finishes. If such an activity exists, it is added to the solution set and the algorithm is called recursively with the new subproblem.


#### Example {#example}

A run of this algorithm is visualized below.

{{< figure src="/ox-hugo/2024-03-24_14-42-29_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Recursive activity selector example (Cormen et al. 2022)." >}}


#### Analysis {#analysis}

At first glance, the algorithm appears to be \\(O(n^2)\\) because of the `while` loop coupled with the recursive call. Once an activity has been selected, the recursive call only considers the activities next \\(n - k\\) activities. The `while` loop picks up where the previous call left off, so the total number of iterations is \\(n\\). The algorithm is \\(O(n)\\).


### Iterative Algorithm {#iterative-algorithm}

The above solution can be adapted to an iterative one.

```python
def greedy_activity_selector(s, f):
    n = len(s) - 1
    A = [1]
    k = 1
    for m in range(2, n + 1):
        if s[m] >= f[k]:
            A.append(m)
            k = m
    return A
```

In the `for` loop, the first line essentially asks if \\(a\_m \in S\_k\\). If so, then add it to the solution set and update \\(k\\) to \\(m\\).


#### Analysis {#analysis}

The analysis of the iterative approach is much clearer. The loop goes over all activities once, so the algorithm is \\(O(n)\\).


## Properties of Greedy Solutions {#properties-of-greedy-solutions}

You can probably imagine a problem for which a greedy solution would not provide the optimal solution. Path planning is one such problem. If we greedily chose the shortest path at each step, we may have missed a shorter path that is not the shortest at each step. The activity selection problem just so happens to be a perfect candidate for a greedy solution, **but what makes it so?**

First, let's review the major steps that led us to the greedy solution for activity selection.

1.  Determine the optimal substructure.
2.  Develop a recursive solution.
3.  Show that making the greedy choice leaves only a single subproblem.
4.  Prove that making the greedy choice leads to an optimal solution.
5.  Develop a recursive algorithm.
6.  Convert it to an iterative algorithm.

The first couple of steps are common to dynamic programming problems. In this case, we could have jumped straight to the greedy approach. Filtering out these extra steps leaves us with:

1.  Cast the optimization problem as one in which we make a choice and are left with a single subproblem.
2.  Prove that the greedy choice is optimal.
3.  Demonstrate optimal substructure: if you make a greedy choice, then you are left with a subproblem such that combining an optimal solution with the greedy choice made previously, you end up with an optimal solution to the original problem.

As we will see, we need two properties to prove that a greedy solution is optimal: the **greedy choice property** and the **optimal substructure property**.


### Greedy Choice Property {#greedy-choice-property}

The **greedy choice property** states that the optimal solution can be found by making locally greedy choices. This approach is opposite of dynamic programming, where the choices at each step are made from the knowledge of optimal solutions to subproblems. That is, dynamic programming is a **bottom-up** approach.

A greedy solution also makes a choice at each step, but it is only based on local information. This is a **top-down** approach. They key of this property is to show that the greedy choice is optimal at each step. For the activity selection problem, the steps were

1.  Examine the optimal solution.
2.  If it has the greedy choice, then the greedy choice is optimal.
3.  If it does not have the greedy choice, then replace the suboptimal choice with the greedy choice.


### Optimal Substructure Property {#optimal-substructure-property}

A problem has optimal substructure if the optimal solution contains optimal solutions to subproblems. We demonstrated this earlier for activity selection. We can start with the assumption that we arrived to a subproblem by making greedy choices. The next step is to show that the optimal solution to the subproblem combined with the greedy choice leads to an optimal solution to the original problem.


### Greedy vs. Dynamic Programming {#greedy-vs-dot-dynamic-programming}

Since there is such overlap between greedy algorithms and dynamic programming in terms of their properties, it is important to understand the differences between the two. To illustrate these difference, we will look at two variations of the same problem.


#### \\(0-1\\) Knapsack Problem {#0-1-knapsack-problem}

Consider the **\\(0-1\\) knapsack problem.**

-   You have \\(n\\) items.
-   Item \\(i\\) is worth \\(v\_i\\) and weights \\(w\_i\\).
-   Find the most valuable subset of items with total weight less than or equal to \\(W\\).
-   Items cannot be divided.

If the most valuable subset of items weighting at most \\(W\\) includes item \\(j\\), then the remaining weight must be the most valuable subset of items weighting at most \\(W - w\_j\\) taken from \\(n-1\\) original items excluding item \\(j\\).


#### Fractional Knapsack Problem {#fractional-knapsack-problem}

This is similar to the \\(0-1\\) knapsack problem, but items can be divided. The objective is to maximize the value of the items in the knapsack.

The optimal substructure of this problem varies slightly: if the most valuable subset weighing at most \\(W\\) includes the weight \\(w\\) of item \\(j\\), then the remaining weight must be the most valuable subset weighing at most \\(W- w\\) that can be taken from the \\(n-1\\) original items plus \\(w\_j - w\\) of item \\(j\\). There is some fraction of item \\(j\\) left after taking \\(w\\) of it.


#### Showing the Greedy Property {#showing-the-greedy-property}

It is established that both problems have optimal substructure. However, only the fractional knapsack problem has the greedy property. Examine the following code:

```python
def fractional_knapsack(v, w, W):
    n = len(v)
    load = 0
    i = 0
    while load < W and i <= n:
        if w[i] <= W - load:
            load += w[i]
            i += 1
        else:
            load += (W - load) * v[i] / w[i]
```

If we are able to sort each item by its value-to-weight ratio, then the greedy choice is to take as much as possible from the most valuable item first, the second most valuable item next, and so on. This considers \\(n\\) items in the worst case, and the items need to be sorted by value-to-weight ratio. The algorithm is \\(O(n \log n)\\).

This does not work for the \\(0-1\\) knapsack problem. Consider the problem visualized below.

{{< figure src="/ox-hugo/2024-03-24_17-05-36_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Greedy solution to the \\(0-1\\) knapsack problem (Cormen et al. 2022)." >}}

In this problem, we have a knapsack whose total capacity is \\(W = 50\\). A table of the weights, values, and value-to-weight ratios is given below.

| \\(i\\)         | 1  | 2   | 3   |
|-----------------|----|-----|-----|
| \\(v\_i\\)      | 60 | 100 | 120 |
| \\(w\_i\\)      | 10 | 20  | 30  |
| \\(v\_i/w\_i\\) | 6  | 5   | 4   |

The fractional algorithm would selection the first item since it has the greatest value-to-weight ratio. The \\(0-1\\) knapsack problem, however, would select the second and third items to maximize the value of the items in the knapsack.
