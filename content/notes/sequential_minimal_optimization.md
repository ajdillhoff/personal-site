+++
title = "Sequential Minimal Optimization"
authors = ["Alex Dillhoff"]
date = 2022-07-04T00:00:00-05:00
tags = ["machine learning"]
draft = false
lastmod = 2026-02-08
sections = "Machine Learning"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Box Constraints](#box-constraints)
- [Updating the Lagrangians](#updating-the-lagrangians)
- [The Algorithm](#the-algorithm)
- [Implementation](#implementation)

</div>
<!--endtoc-->



## Introduction {#introduction}

**Paper link:** <https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/>

Sequential Minimal Optimization (SMO) is an algorithm to solve the SVM Quadratic Programming (QP) problem efficiently (<a href="#citeproc_bib_item_1">Platt 1998</a>). Developed by John Platt at Microsoft Research, SMO deals with the constraints of the SVM objective by breaking it down into a smaller optimization problem at each step.

The two key components of SMO are

1.  an analytic method to solving for two Lagrange multipliers at a time
2.  and a heuristic for choosing which multipliers to optimize.

The original objective is to maximize the margin between the nearest positive and negative examples.
For the linear case, if the output is given as

\\[
u = \mathbf{w}^T \mathbf{x} - b,
\\]

where \\(\mathbf{w}\\) is the normal vector to the hyperplane separating the classes, then the margin is given as

\\[
m = \frac{1}{\\|w\\|\_2}.
\\]

Maximizing this margin yielded the primal optimization problem

\begin{align\*}
\min\_{\mathbf{w},b} \frac{1}{2} \\|\mathbf{w}\\|^2\\\\
\textrm{s.t.} \quad & y\_i(\mathbf{w}^T \mathbf{x} - b) \geq 1, \forall i\\\\
\end{align\*}

The dual form of the objective function for a [Support Vector Machine]({{< relref "support_vector_machine.md" >}}) is

\\[
\min\_{\vec\alpha} \Psi(\vec{\alpha}) = \min\_{\vec{\alpha}} \frac{1}{2}\sum\_{i=1}^N \sum\_{j=1}^N y\_i y\_j K(\mathbf{x}\_i, \mathbf{x}\_j)\alpha\_i\alpha\_j - \sum\_{i=1}^N \alpha\_i
\\]

with inequality constraints

\\[
\alpha\_i \geq 0, \forall i,
\\]

and a linear equality constraint

\\[
\sum\_{i=1}^N y\_i \alpha\_i = 0.
\\]

For a linear SVM, the output is dependent on a weight vector \\(\mathbf{w}\\) and threshold \\(b\\):

\\[
\mathbf{w} = \sum\_{i=1}^N y\_i \alpha\_i \mathbf{x}\_i, \quad b = \mathbf{w}^T \mathbf{x}\_k - y\_k.
\\]

****The threshold is also dependent on the weight vector?**** The weight vector \\(\mathbf{w}\\) is computed using the training data. The threshold is only dependent on non-zero support vectors, \\(\alpha\_k > 0\\).


### Overlapping Distributions {#overlapping-distributions}

Slack variables were introduced to allow misclassifications at the cost of a linear penalty.
This is useful for datasets that are not linearly separable.
In practice, this is accomplished with a slight modification of the original objective function:

\begin{align\*}
\min\_{\mathbf{w},b} \frac{1}{2} \\|\mathbf{w}\\|^2 + C \sum\_{i=1}^N \xi\_i\\\\
\textrm{s.t.} \quad & y\_i(\mathbf{w}^T \mathbf{x} - b) \geq 1 - \xi\_i, \forall i\\\\
\end{align\*}

The convenience of this formulation is that the parameters \\(\xi\_i\\) do not appear in the dual formulation at all.
The only added constraint is

\\[
0 \leq \alpha\_i \leq C, \forall i.
\\]

This is referred to as the box constraint for reasons we shall see shortly.


## Box Constraints {#box-constraints}

The smallest optimization step that SMO solves is that of two variables.
Given the constraints above, the solution lies on a diagonal line \\(\sum\_{i=1}^N y\_i \alpha\_i = 0\\) bounded within a box \\(0 \leq \alpha\_i \leq C, \forall i\\).

Isolating for two samples with alphas \\(\alpha\_1\\) and \\(\alpha\_2\\), the constraint \\(\sum\_{i=1}^n y\_i \alpha\_i = 0\\) suggests that

\\[
y\_1 \alpha\_1 + y\_2 \alpha\_2 = w.
\\]

We first consider the case when \\(y\_1 \neq y\_2\\).
Let \\(y\_1 = 1\\) and \\(y\_2 = -1\\), then \\(a\_1 - a\_2 = w\\).
As \\(\alpha\_1\\) increases, \\(\alpha\_2\\) must also increase to satisfy the constraint.

{{< figure src="/ox-hugo/2022-07-10_22-48-56_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Equality constraint for case 1 (<a href=\"#citeproc_bib_item_1\">Platt 1998</a>)." >}}

The other case is when \\(y\_1 = y\_2\\), then \\(\alpha\_1 + \alpha\_2 = w\\).
As \\(\alpha\_1\\) is increased, \\(\alpha\_2\\) is decreased to satisfy the constraint.

{{< figure src="/ox-hugo/2022-07-10_22-51-53_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Box constraint for samples of the same class (<a href=\"#citeproc_bib_item_1\">Platt 1998</a>)." >}}


## Updating the Lagrangians {#updating-the-lagrangians}

SMO solves for only two Lagrange multipliers at a time.
Solving for only 1 at a time would be impossible under the constraint \\(\sum\_{i=1}^N y\_i \alpha\_i = 0\\).
The first step is to compute \\(\alpha\_2\\) and constrain it between the ends of the diagonal line segment from the box constraints.

If \\(y\_1 \neq y\_2\\), then the following bounds are applied to \\(\alpha\_2\\):

\begin{equation\*}
L = \max(0, \alpha\_2 - \alpha\_1), \quad H = \min(C, C + \alpha\_2 - \alpha\_1)
\end{equation\*}

otherwise, the bounds are computed as:

\begin{equation\*}
L = \max(0, \alpha\_2 + \alpha\_1 - C), \quad H = \min(C, \alpha\_2 + \alpha\_1)
\end{equation\*}

Updating the actual parameter is done following the update rule of gradient descent:

\\[
\alpha\_2^{\text{new}} = \alpha\_2 + \frac{y\_2(E\_1 - E\_2)}{\eta}.
\\]

**How do we arrive at this update rule?**


### Second Derivative of the Objective Function {#second-derivative-of-the-objective-function}

Here, \\(\eta\\) represents the step size and direction. It is computed from the second derivative of the objective function along the diagonal line. To see that this is the case, consider the original objective function

\begin{align\*}
\min\_{\mathbf{\alpha}} \quad & \frac{1}{2} \sum\_{i=1}^N \sum\_{j=1}^N y\_i y\_j K(\mathbf{x}\_i, \mathbf{x}\_j) \mathbf{\alpha}\_1 \mathbf{\alpha}\_2 - \sum\_{i=1}^N \alpha\_i\\\\
\textrm{s.t.} \quad & 0 \leq \alpha\_i \leq C, \forall i\\\\
                    & \sum\_{i=1}^N y\_i \alpha\_i = 0\\\\
\end{align\*}

Since we are optimizing with respect to only 2 Lagrangian multipliers at a time, we can write the Lagrangian function as

\\[
\frac{1}{2} y\_1^2 K\_{11} \alpha\_1^2 + \frac{1}{2} y\_2^2 K\_{22} \alpha\_2^2 + y\_1 \alpha\_1 \sum\_{j=3}^N y\_j \alpha\_j K\_{1j} + y\_2 \alpha\_2 \sum\_{j=3}^N y\_j \alpha\_j K\_{2j} - \alpha\_1 - \alpha\_2 + \sum\_{j=3}^N \alpha\_j
\\]

We are only optimizing with respect to \\(\alpha\_1\\) and \\(\alpha\_2\\), the next step is to extract those terms from the sum.
This is simplified further by noting that \\(\sum\_{j=3}^N y\_j \alpha\_j K\_{ij}\\) looks very similar to the output of an SVM:

\\[
u = \sum\_{j=1}^N y\_j \alpha\_j K(\mathbf{x}\_j, \mathbf{x}) - b.
\\]

This allows us to introduce a variable \\(v\_i\\) based on \\(u\_i\\), the output of an SVM given sample \\(\mathbf{x}\_i\\):

\\[
v\_i = \sum\_{j=3}^N y\_j \alpha\_j K\_{ij} = u\_i + b - y\_1 \alpha\_1 K\_{1i} - y\_2 \alpha\_2 K\_{2i}.
\\]

The objective function is then written as

\\[
\frac{1}{2} y\_1^2 K\_{11} \alpha\_1^2 + \frac{1}{2} y\_2^2 K\_{22} \alpha\_2^2 + y\_1 \alpha\_1 v\_1 + y\_2 \alpha\_2 v\_2 - \alpha\_1 - \alpha\_2 + \sum\_{j=3}^N \alpha\_j.
\\]

Note that the trailing sum \\(\sum\_{j=3}^N \alpha\_j\\) is treated as a constant since those values are not considered when optimizing for \\(\alpha\_1\\) and \\(\alpha\_2\\).

Given the box constraints from above, we must update \\(\alpha\_1\\) and \\(\alpha\_2\\) such that

\\[
\alpha\_1 + s \alpha\_2 = \alpha\_1^\* + s \alpha\_2^\* = w.
\\]

This linear relationship allows us to express the objective function in terms of &alpha;_2:

\\[
\Psi = \frac{1}{2} y\_1^2 K\_{11} (w - s \alpha\_2)^2 + \frac{1}{2} y\_2^2 K\_{22} \alpha\_2^2 + y\_1 (w - s \alpha\_2) v\_1 + y\_2 \alpha\_2 v\_2 - \alpha\_1 - \alpha\_2 + \sum\_{j=3}^N \alpha\_j.
\\]

The extremum of the function is given by the first derivative with respect to \\(\alpha\_2\\):

\\[
\frac{d\Psi}{d\alpha\_2} = -sK\_{11}(w - s\alpha\_2) + K\_{22}\alpha\_2 - K\_{12}\alpha\_2 + s K\_{12} (w - s \alpha\_2) - y\_2 v\_2 + s + y\_2 v\_2 - 1 = 0.
\\]

In most cases, the second derivative will be positive.
The minimum of \\(\alpha\_2\\) is where

\begin{align\*}
\alpha\_2 (K\_{11} + K\_{22} - 2 K\_{12}) &= s(K\_{11} - K\_{12})w + y\_2(v\_1 - v\_2) + 1 - s\\\\
&= s(K\_{11} - K\_{12})(s\alpha\_2^\*+\alpha\_1^\*)\\\\
&+ y\_2(u\_1-u\_2+y\_1\alpha\_1^\*(K\_{12} - K\_{11}) + y\_2 \alpha\_2^\* (K\_{22} - K\_{21})) + y\_2^2 - s\\\\
&= \alpha\_2^\*(K\_{11}+K\_{22} - 2K\_{12}) + y\_2(u\_1 - u\_2 + y\_2 - y\_1).
\end{align\*}

If we let \\(E\_1 = u\_1 - y\_1\\), \\(E\_2 = u\_2 - y\_2\\), and \\(\eta = K\_{11} + K\_{22} - 2K\_{12}\\), then

\\[
\alpha\_2^{\text{new}} = \alpha\_2 + \frac{y\_2(E\_1 - E\_2)}{\eta}.
\\]


## The Algorithm {#the-algorithm}

Sequential Minimal Optimization (SMO) solves the SVM problem which usually requires a Quadratic Programming (QP) solution.
It does this by breaking down the larger optimization problem into a small and simple form: solving for two Lagrangians.
Solving for one would not be possible without violating KKT conditions.
There are two components to Sequential Minimal Optimization: the first is how the Lagrangians are selected and the second is the actual optimization step.


### Choosing the First Lagrangian {#choosing-the-first-lagrangian}

The algorithm first determines which samples in the dataset violate the given KKT conditions. Only those violating the conditions are eligible for optimization. A solution is found when the following are true for all \\(i\\):

\begin{align\*}
\alpha\_i = 0 \iff y\_i u\_i \geq 1,\\\\
0 < \alpha\_i < C \iff y\_i u\_i = 1,\\\\
\alpha\_i = C \iff y\_i u\_i \leq 1.
\end{align\*}

Additionally, samples that are not on the bounds are selected (those with \\(\alpha\_i \neq 0\\) and \\(\alpha\_i \neq C\\)).
This continues through the dataset until no sample violates the KKT constraints within \\(\epsilon\\).

As a last step, SMO searches the entire dataset to look for any bound samples that violate KKT conditions. It is possible that updating a non-bound sample would cause a bound sample to violate the KKT conditions.


### Choosing the Second Lagrangian {#choosing-the-second-lagrangian}

The second Lagrangian is chosen to maximize the size of the step taken during joint optimization.
Noting that the step size is based on

\\[
\alpha\_2^{\text{new}} = \alpha\_2 + \frac{y\_2(E\_1 - E\_2)}{\eta},
\\]

it is approximated by computing \\(|E\_1 - E\_2|\\).

If positive progress cannot be made given the choice of Lagrangian, SMO will begin iterating through non-bound examples.
If no eligible candidates are found in the non-bound samples, the entire dataset is searched.


### Updating the Parameters {#updating-the-parameters}

With the second derivative of the objective function, we can take an optimization step along the diagonal line.
To ensure that this step adheres to the box constraints defined above, the new value of \\(\alpha\_2\\) is clipped:

\begin{equation\*}
\alpha\_2^{\text{new,clipped}} =
\begin{cases}
H &\text{if} \quad \alpha\_2^{\text{new}} \geq H;\\\\
\alpha\_2^{\text{new}} &\text{if} \quad L < \alpha\_2^{\text{new}} < H;\\\\
L &\text{if} \quad \alpha\_2^{\text{new}} \geq L.\\\\
\end{cases}
\end{equation\*}

With the new value of \\(\alpha\_2\\), \\(\alpha\_1\\) is computed such that the original KKT condition is preserved:

\\[
\alpha\_1^{\text{new}} = \alpha\_1 + s(\alpha\_2 - \alpha\_2^{\text{new,clipped}}),
\\]

where \\(s = y\_1y\_2\\).

Points that are beyond the margin are given an alpha of 0: \\(\alpha\_i = 0\\).
Points that are on the margin satisfy \\(0 < \alpha\_i < C\\). These are the support vectors.
Points inside the margin satisfy \\(\alpha\_i = C\\).


#### Linear SVMs {#linear-svms}

In the case of linear SVMs, the parameters can be stored as a single weight vector

\\[
\mathbf{w}^{\text{new}} = \mathbf{w} + y\_1 (\alpha\_1^{\text{new}} - \alpha\_1)\mathbf{x}\_1 + y\_2(\alpha\_2^{\text{new,clipped}} - \alpha\_2)\mathbf{x}\_2.
\\]

The output of a linear SVM is computed as

\\[
u = \mathbf{w}^T \mathbf{x} - b.
\\]


#### Nonlinear SVMs {#nonlinear-svms}

In the nonlinear case, the output of the model is computed as

\\[
u = \sum\_{i=1}^N y\_i \alpha\_i K(\mathbf{x}\_i, \mathbf{x}) - b.
\\]


#### Computing the bias {#computing-the-bias}

There is yet one parameter left to update: this bias term. Given the equation for SVM's output above, we can compute the output as

\\[
b = \mathbf{w}^T \mathbf{x}\_k - y\_k\quad \text{for } \alpha\_k > 0.
\\]

We can also come up with an equation for updating \\(b\\) intuitively. The support vectors lie in the margins, where \\(\alpha\_i = 1\\). That is, where \\(y\_i \mathbf{w}^T \mathbf{x}\_i = 1\\). Then taking the average of a support vector lying on each margin yields \\(b\\).

\\[
b = - \frac{1}{2} \Big( \max\_{i:y\_i=-1} \mathbf{w}^T \mathbf{x}\_i + \min\_{i:y\_i=1} \mathbf{w}^T \mathbf{x}\_i \Big).
\\]

{{< notice "note" "SMO's Iterative Solution" >}}
The updates given above assume that the optimal solution for $\mathbf{w}$ has already been found. The bias update term given by Platt allows for iterative updates. See the paper for more details.
{{< /notice >}}


## Implementation {#implementation}

An implementation of SMO in Python is available at <https://github.com/ajdillhoff/CSE6363/blob/main/svm/smo.ipynb>

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Platt, John C. 1998. “Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,” 21.</div>
</div>
