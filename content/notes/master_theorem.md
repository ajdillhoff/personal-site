+++
title = "Master Theorem"
authors = ["Alex Dillhoff"]
date = 2024-02-04T17:49:00-06:00
tags = ["computer science", "algorithms"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Example: Merge Sort](#example-merge-sort)
- [Example: Matrix Multiplication](#example-matrix-multiplication)
- [Example: Median Finding](#example-median-finding)
- [Example: Cormen et al. Exercise 4.5-2](#example-cormen-et-al-dot-exercise-4-dot-5-2)

</div>
<!--endtoc-->

In the study of [Divide and Conquer Algorithms]({{< relref "divide_and_conquer_algorithms.md" >}}), a recurrence tree can be used to determine the runtime complexity. These notes focus on the **master theorem**, a blueprint for solving any recurrence of the form

\\[
T(n) = aT(n/b) + f(n).
\\]

-   \\(n\\) is the size of the problem,
-   \\(a \geq 1\\) is the number of subproblems,
-   \\(b > 1\\) is the factor by which the problem size is reduced, and
-   \\(f(n)\\) is the cost of the work done outside of the recursive calls.

Each recurrence is solved in \\(T(n/b)\\) time, and \\(f(n)\\) would include the cost of dividing and recombining the problem. The full theorem as described in _Introduction to Algorithms_ is restated below (<a href="#citeproc_bib_item_1">Cormen et al. 2022</a>).

****Master Theorem****

Let \\(a > 0\\) and \\(b > 1\\) be constants, and let \\(f(n)\\) be a driving function that is defined and nonnegative on all sufficiently large reals. Define the recurrence \\(T(n)\\) on \\(n \in \mathbb{N}\\) by

\\[
T(n) = aT(n/b) + f(n),
\\]

where \\(aT(n/b)\\) actually means \\(a'T(\lfloor n/b \rfloor) + a{''}T(\lceil n / b \rceil)\\) for some constants \\(a' \geq 0\\) and \\(a'' \geq 0\\) such that \\(a = a' + a''\\). Then \\(T(n)\\) has the following asymptotic bounds:

1.  If \\(f(n) = O(n^{\log\_b a - \epsilon})\\) for some constant \\(\epsilon > 0\\), then \\(T(n) = \Theta(n^{\log\_b a})\\).
2.  If \\(f(n) = \Theta(n^{\log\_b a} \log^k n)\\) for some constant \\(k \geq 0\\), then \\(T(n) = \Theta(n^{\log\_b a} \log^{k+1} n)\\).
3.  If \\(f(n) = \Omega(n^{\log\_b a + \epsilon})\\) for some constant \\(\epsilon > 0\\), and if \\(a f(n/b) \leq k f(n)\\) for some constant \\(k < 1\\) and all sufficiently large \\(n\\), then \\(T(n) = \Theta(f(n))\\).


### Theorem Breakdown {#theorem-breakdown}

The common function \\(n^{\log\_b a}\\) is called the **watershed function**. The driving function \\(f(n)\\) is compared to it to determine which case applies. If the watershed function grows at a faster rate than \\(f(n)\\), case 1 applies. If they grow at the same rate, case 2 applies. If \\(f(n)\\) grows at a faster rate, case 3 applies.

In case 1, the watershed function should grow faster than \\(f(n)\\) by a factor of \\(n^\epsilon\\) for some \\(\epsilon > 0\\). In case 2, technically the watershed function should grow at least the same rate as \\(f(n)\\), if not faster. That is, it grows faster by a factor of \\(\Theta(\log^k n)\\), where \\(k \geq 0\\). You can think of the extra \\(\log^k n\\) as an augmentation to the watershed function to ensure that they grow at the same rate. In most cases, \\(k = 0\\) which results in \\(T(n) = \Theta(n^{\log\_b a} \log n)\\).

Since case 2 allows for the watershed function to grow faster than \\(f(n)\\), case 3 requires that it grow at least **polynomially** faster. \\(f(n)\\) should grow faster by at least a factor of \\(\Theta(n^\epsilon)\\) for some \\(\epsilon > 0\\). Additionally, the driving function must satisfy the regularity condition \\(a f(n/b) \leq k f(n)\\) for some constant \\(k < 1\\) and all sufficiently large \\(n\\). This condition ensures that the cost of the work done outside of the recursive calls is not too large.


### Application of the Master Method {#application-of-the-master-method}

In most cases, the master method can be applied by looking at the recurrence and applying the relevant case. If the driving and watershed functions are not immediately obvious, you can use a different method as discussed in [Divide and Conquer Algorithms]({{< relref "divide_and_conquer_algorithms.md" >}}).

When this can be applied, it is much simpler than the other methods. Let's revisit some of the main problems we've explored before discussing applications for which the master method could not be used.


## Example: Merge Sort {#example-merge-sort}

Merge Sort has a recurrence of the form \\(T(n) = 2T(n/2) + \Theta(n)\\). The driving function is \\(f(n) = \Theta(n)\\). The constants \\(a\\) and \\(b\\) are both 2, so the watershed function is \\(n^{\log^2 2}\\), which is \\(n\\). Since \\(f(n)\\) grows at the same rate as the watershed function, case 2 applies. Therefore, \\(T(n) = \Theta(n \log n)\\).


## Example: Matrix Multiplication {#example-matrix-multiplication}

The recurrence of the divide and conquer version of matrix multiplication for square matrices is \\(T(n) = 8T(n/2) + \Theta(1)\\). Given \\(a = 8\\) and \\(b = 2\\), we can see that the complexity is inherent in the recurrence, not the driving function. The watershed function is \\(n^{\log\_2 8}\\), which is \\(n^3\\). This grows at a faster rate than \\(\Theta(1)\\), so case 1 applies. Therefore, \\(T(n) = \Theta(n^3)\\).


## Example: Median Finding {#example-median-finding}

Median finding has a recurrence of the form \\(T(n) = T(n/5) + T(7n/10) + \Theta(n)\\). Given the two recurrence factors, how do we evaluate the driving function? The form itself does not fit the master theorem, so it cannot be applied in this case. We could use the substitution method, recurrence trees, or the Akra-Bazzi theorem to solve this one.


## Example: Cormen et al. Exercise 4.5-2 {#example-cormen-et-al-dot-exercise-4-dot-5-2}

In this exercise from _Introduction to Algorithms_, we are asked to find the largest integer value \\(a\\) such that an algorithm with the recurrence \\(T(n) = aT(n/4) + \Theta(n^2)\\) is asymptotically faster than \\(\Theta(n^{\log\_2 7})\\). Since \\(b = 4\\), the largest integer \\(a\\) will be the smallest integer such that \\(\log\_4 a < \log\_2 7\\). Solving for the inequality shows that \\(a = 48\\) is the largest such integer.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Cormen, Thomas H., Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2022. <i>Introduction to Algorithms</i>. 4th ed. MIT Press. <a href="http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/">http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/</a>.</div>
</div>
