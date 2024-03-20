+++
title = "Recursion Tree Method"
authors = ["Alex Dillhoff"]
date = 2024-03-18T22:10:00-05:00
tags = ["computer science", "algorithms"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Example 4.13 from CLRS](#example-4-dot-13-from-clrs)

</div>
<!--endtoc-->

Visualizing the characteristics of an algorithm is a great way to build intuition about its runtime. Although it can be used to prove a recurrence, it is often a good jumping off point for the [Substitution Method]({{< relref "substitution_method.md" >}}).


## Example 4.13 from CLRS {#example-4-dot-13-from-clrs}

Consider the recurrence \\(T(n) = 3T(n/4) + \Theta(n^2)\\). We start by describing \\(\Theta(n^2) = cn^2\\), where the constant \\(c > 0\\) serves as an upper-bound constant. It reflects the amount of work done at each level of the recursion tree. The tree is shown below.

{{< figure src="/ox-hugo/2024-03-19_09-14-00_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Example 4.13 from CLRS (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)" >}}

As the tree expands out over a few levels, we can see a pattern in the cost at depth \\(i\\). Each level of increasing depth has 3 times as many nodes as the previous. With the exception of the leaves, the cost for each level is \\((\frac{3}{16})^i cn^2\\). The total cost of the leaves is based on the number of leaves, which is \\(3^{\log\_4 n}\\) since each level has \\(3^i\\) nodes and the depth is \\(\log\_4 n\\). Using the identity \\(a^{\log\_b c} = c^{\log\_b a}\\), we can simplify the leaves to \\(n^{\log\_4 3}\\). The total cost of the leaves is \\(\Theta(n^{\log\_4 3})\\).

The last step is to add up the costs over all levels:

\begin{align\*}
T(n) &= \sum\_{i=0}^{\log\_4 n} \left( \frac{3}{16} \right)^i cn^2 + \Theta(n^{\log\_4 3}) \\\\
&< \sum\_{i=0}^{\infty} \left( \frac{3}{16} \right)^i cn^2 + \Theta(n^{\log\_4 3}) \\\\
&= \frac{cn^2}{1 - \frac{3}{16}} + \Theta(n^{\log\_4 3}) \\\\
&= \frac{16}{13}cn^2 + \Theta(n^{\log\_4 3}) \\\\
&= \Theta(n^2).
\end{align\*}

The second line in the equation is a [geometric series.](https://en.wikipedia.org/wiki/Geometric_series)


### Verifying using the Substitution Method {#verifying-using-the-substitution-method}

Even if we weren't so particular with the maths, the recursion tree method is a great way to build intuition about the runtime. Let's verify that this recurrence is bounded above by \\(O(n^2)\\) using the [Substitution Method]({{< relref "substitution_method.md" >}}).

Here we show that \\(T(n) \leq dn^2\\) for a constant \\(d > 0\\). The previous constant \\(c > 0\\) is reused to describe the cost at each level of the recursion tree.

\begin{align\*}
T(n) &\leq 3T(n/4) + cn^2 \\\\
&\leq 3(d(n/4)^2) + cn^2 \\\\
&= \frac{3}{16}dn^2 + cn^2 \\\\
&\leq dn^2 \text{ if } d \geq \frac{16}{13}c.
\end{align\*}


### Exercises {#exercises}

1.  Solve the recurrence \\(T(n) = 2T(n/2) + cn\\) using the recursion tree method.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Cormen, Thomas H., Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2022. <i>Introduction to Algorithms</i>. 4th ed. MIT Press. <a href="http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/">http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/</a>.</div>
</div>
