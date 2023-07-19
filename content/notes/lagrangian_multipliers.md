+++
title = "Lagrangian Multipliers"
authors = ["Alex Dillhoff"]
date = 2022-02-05T00:00:00-06:00
draft = false
+++

## Introduction {#introduction}

Let's take a simple constrained problem (from Nocedal and Wright).

\begin{align\*}
    \min \quad & x\_1 + x\_2\\\\
\textrm{s.t.} \quad & x\_1^2 + x\_2^2 - 2 = 0
\end{align\*}

The set of possible solutions to this problem lie on the boundary of the circle defined by the constraint:

{{< figure src="/ox-hugo/2021-12-01_18-06-50_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Source: Nocedal and Wright" >}}

If we let \\(g(\mathbf{x}) = x\_1^2 + x\_2^2 - 2\\), then the gradient vector is \\((2x\_1, 2x\_2)\\)

Our original function \\(f(\mathbf{x}) = x\_1 + x\_2\\) has a gradient vector of \\((1, 1)\\).

The figure above visualizes these vectors at different points on the constraint boundary.

Notice that the optimal solution \\(\mathbf{x}^\* = (-1, -1)\\) is at a point where \\(\nabla g(\mathbf{x}^\*)\\) is parallel to \\(\nabla f(\mathbf{x}^\*)\\). However, the gradients of the vectors are not equal. So there must be some scalar \\(\lambda\\) such that \\(\nabla f(\mathbf{x}^\*) = \lambda \nabla g(\mathbf{x}^\*)\\).

This scalar \\(\lambda\\) is called a Lagrangian multiplier. We use this and introduce the Lagrangian function:

\begin{equation\*}
    \mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})
\end{equation\*}

This yields a form for which we can analytically calculate the stationary points. That is,

\begin{equation\*}
    \nabla\_{\mathbf{x}} \mathcal{L}(\mathbf{x}^\*, \lambda^\*) = 0.
\end{equation\*}


## Lagrangian Duality {#lagrangian-duality}

In general, the primal optimization problem is formulated as

\begin{align\*}
\min\_{w} \quad & f(w)\\\\
\textrm{s.t.} \quad & g\_i(w) \leq 0, \quad i = 1, \dots, k\\\\
& h\_i(w) = 0, \quad i = 1, \dots, l.
\end{align\*}

The Lagrangian function is then

\\[
L(w, \alpha, \beta) = f(w) + \sum\_{i=1}^k\alpha\_i g\_i(w) + \sum\_{i=1}^l \beta\_i h\_i(w).
\\]


## Additional Resources {#additional-resources}

-   <https://cs229.stanford.edu/notes2021fall/cs229-notes3.pdf>
