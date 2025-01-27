+++
title = "Support Vector Machine"
authors = ["Alex Dillhoff"]
date = 2022-01-22T00:00:00-06:00
tags = ["machine learning"]
draft = false
lastmod = 2025-01-26
sections = "Machine Learning"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Maximum Margin Classifier](#maximum-margin-classifier)
- [Formulation](#formulation)
- [Overlapping Class Distributions](#overlapping-class-distributions)
- [Multiclass SVM](#multiclass-svm)
- [Additional Resources](#additional-resources)

</div>
<!--endtoc-->



## Introduction {#introduction}

Support Vector Machines are a class of supervised learning methods primarily used for classification. Although they can be formulated for regression and outlier detection as well. Instead of optimizing a set of parameters which compress or summarize the training set, they use a small subset of the training data to compute the decision function.

They rely on the data being linearly separable, so feature transformations are critical for problems in which the original representation of the data is not linearly separable.


## Maximum Margin Classifier {#maximum-margin-classifier}

Let's start with a simple classification model as we studied with [Logistic Regression]({{< relref "logistic_regression.md" >}}). That is, we have

\\[
f(\mathbf{x}) = \mathbf{w}^T\phi(\mathbf{x}),
\\]

where \\(\phi(\mathbf{x})\\) is a function which transforms our original input into some new feature space. The transformed input is assumed to be linearly separable so that a decision boundary can be computed. In the original logistic regression problem, a decision boundary was found through optimization. For linearly separable data, there are an infinite number of decision boundaries that satisfy the problem.

**What about the quality of the decision boundary?**

Is one decision boundary better than the other?


## Formulation {#formulation}

Given a training set \\(\\{\mathbf{x}\_1, \dots, \mathbf{x}\_n\\}\\) with labels \\(\\{y\_1, \dots, y\_n\\}\\), where \\(y\_i \in \\{-1, 1\\}\\), we construct a linear model which classifies an input sample depending on the sign of the output.

Our decision rule for classification, given some input \\(\mathbf{x}\\), is

\begin{equation\*}
f(\mathbf{x}) =
\begin{cases}
1\text{ if }\mathbf{w}^T\mathbf{x} + b \geq 0\\\\
-1\text{ if }\mathbf{w}^T\mathbf{x} + b < 0
\end{cases}
\end{equation\*}

**How large should the margin be?**

In the original formulation of [Logistic Regression]({{< relref "logistic_regression.md" >}}), we saw that the parameter vector \\(\mathbf{w}\\) described the **normal** to the decision boundary. The distance between a given point \\(\mathbf{x}\\) and the decision boundary is given by

\\[
\frac{y\_if(\mathbf{x})}{||\mathbf{w}||}.
\\]

We can frame this as an optimization problem: come up with a value for \\(\mathbf{w}\\) that maximizes the margin.

\\[
\text{arg max}\_{\mathbf{w}, b} \frac{1}{\\|\mathbf{w}\\|}\min\_{i} y\_i (\mathbf{w}^T\phi(\mathbf{x}\_i) + b)
\\]

We can arbitrarily scale the parameters, so we add an additional constraint that any point that lies on the boundary of the margin satisfies

\\[
y\_i(\mathbf{w}^T\mathbf{x} + b) = 1.
\\]

Under this constraint, we have that all samples satisfy

\\[
y\_i(\mathbf{w}^T\mathbf{x} + b) \geq 1.
\\]

That is, all positive samples with target \\(1\\) will produce at least a \\(1\\), yielding a value greater than or equal to 1. All negative samples with target \\(-1\\) will produce at most a \\(-1\\), yielding a value greater than or equal to 1.

Another way of writing this is

\begin{equation\*}
f(\mathbf{x}) =
\begin{cases}
1\text{ if }\mathbf{w}^T\mathbf{x}\_{+} + b \geq 1\\\\
-1\text{ if }\mathbf{w}^T\mathbf{x}\_{-} + b \leq -1,
\end{cases}
\end{equation\*}

where \\(\mathbf{x}\_+\\) is a positive sample and \\(\mathbf{x}\_-\\) is a negative sample. The decision rule can then be written as

\\[
y\_i(\mathbf{w}^T\mathbf{x} + b) - 1 \geq 0.
\\]

This implies that the only samples that would yield an output of 0 are those that lie directly on the margins of the decision boundary.

Given this constraint of \\(y\_i(\mathbf{w}^T\mathbf{x} + b) - 1 = 0\\), we can derive our optimization objective.

The margin can be computed via the training data. To do this, consider two data points which lie on their respective boundaries, one positive and one negative, and compute the distance between them: \\(\mathbf{x}\_+ - \mathbf{x}\_-\\). This distance with respect to our decision boundary, defined by \\(\mathbf{w}\\), is given by

\\[
(\mathbf{x}\_+ - \mathbf{x}\_-) \cdot \frac{\mathbf{w}}{||\mathbf{w}||}.
\\]

For clarity, we can rewrite this as

\begin{equation\*}
\frac{1}{||\mathbf{w}||}(\mathbf{x}\_{+} \cdot \mathbf{w} - \mathbf{x}\_{-} \cdot \mathbf{w}).
\end{equation\*}

If we substitute the sample values into the equality constraint above, we can simplify this form. For the positive sample, we have \\(\mathbf{w}^T\mathbf{x} = 1 - b\\). For the negative sample, we get \\(\mathbf{w}^T\mathbf{x} = -1 - b\\). The equation above then becomes

\begin{equation\*}
\frac{1}{||\mathbf{w}||}(1 - b - (-1 - b)) = \frac{2}{||\mathbf{w}||}.
\end{equation\*}

Thus, our objective is to maximize \\(\frac{2}{||\mathbf{w}||}\\) which is equivalent to minimizing \\(\frac{1}{2}||\mathbf{w}||^2\\) subject to the constraints \\(y\_i(\mathbf{w}^T\mathbf{x}+b)\geq 1\\). This is a constrainted optimization problem. As discussed previously, we can simplify such problems by introducing [Lagrangian Multipliers]({{< relref "lagrangian_multipliers.md" >}}). Doing this produces the dual representation of our optimization objection:

\begin{equation\*}
L = \frac{1}{2}||\mathbf{w}||^2 - \sum\_{i=1}^n \alpha\_i \big(y\_i(\mathbf{w}^T\mathbf{x}\_i + b) - 1\big).
\end{equation\*}

To solve for \\(\mathbf{w}\\) we compute \\(\frac{\partial}{\partial \mathbf{w}}L\\).

\begin{equation\*}
\frac{\partial}{\partial \mathbf{w}}L = \mathbf{w} - \sum\_{i=1}^n \alpha\_i y\_i \mathbf{x}\_i.
\end{equation\*}

Setting this to 0 yields

\begin{equation\*}
\mathbf{w} = \sum\_{i=1}^n \alpha\_i y\_i \mathbf{x}\_i.
\end{equation\*}

Doing the same for the other parameter \\(b\\) yields

\\[
0 = \sum\_{i=1}^n \alpha\_i y\_i.
\\]

We can now simplify our objective function by substituting these results into it:

\begin{align\*}
L &= \frac{1}{2}\Big(\sum\_{i=1}^n \alpha\_i y\_i \mathbf{x}\_i\Big)^2 - \sum\_{i=1}^n \alpha\_i\Big(y\_i\big((\sum\_{i=1}^n\alpha\_i y\_i \mathbf{x}\_i)^T\mathbf{x}\_i + b \big) - 1 \Big)\\\\
&= \frac{1}{2}\Big(\sum\_{i=1}^n \alpha\_i y\_i \mathbf{x}\_i\Big)^2 - \Big(\sum\_{i=1}^n \alpha\_i y\_i \mathbf{x}\_i \Big)^2 - \sum\_{i=1}^n \alpha\_i y\_i b + \sum\_{i=1}^n \alpha\_i\\\\
&= -\frac{1}{2} \Big(\sum\_{i=1}^n \alpha\_i y\_i \mathbf{x}\_i \Big)^2 + \sum\_{i=1}^n \alpha\_i\\\\
&= \sum\_{i=1}^n \alpha\_i - \frac{1}{2}\sum\_{i=1}^n\sum\_{j=1}^m \alpha\_i \alpha\_j y\_i y\_j \mathbf{x}\_i \cdot \mathbf{x}\_j
\end{align\*}

Thus, the objective is dependent on the inner product of samples \\(\mathbf{x}\_i\\) and \\(\mathbf{x}\_j\\). If these were representations in some complex feature space, our problem would remain computationally inefficient. However, we can take advantage of [Kernels]({{< relref "kernels.md" >}}) for this.

Note that, in most cases, \\(\alpha\_i\\) will be 0 since we only consider **support vectors**. That is, the points that lie on the margins of the decision boundary.


## Overlapping Class Distributions {#overlapping-class-distributions}

The above formulation is fine and works with datasets that have no overlap in feature space.
That is, they are completely linearly separable.
However, it is not always the case that they will be.

To account for misclassifications while still maximizing a the margin between datasets, we introduce a penalty value for points that are misclassified.
As long as there aren't too many misclassifications, this penalty will stay relatively low while still allowing us to come up with an optimal solution.

This penalty comes in the form of a **slack variable** \\(\xi\_i \geq 0\\) for each sample that is \\(0\\) for points that are on or inside the correct margin and \\(\xi\_i = |y\_i - f(\mathbf{x})|\\) for others.
If the point is misclassified, its slack variable will be \\(\xi\_i > 1\\).


## Multiclass SVM {#multiclass-svm}

Similar to our simple [Logistic Regression]({{< relref "logistic_regression.md" >}}) method, SVMs are binary classifiers by default. We can take a similar approach to extending them to multiple classes, but there are downsides to each approach.

The "one-vs-all" approach entails building \\(|K|\\) classifiers and choose the classifier which predicts the input with the greatest margin.

The "one-vs-one" approach involves building \\(|K|\cdot\frac{|K| - 1}{2}\\) classifiers. In this case, training each classifer will be more tractable since the amount of data required for each one is less. For example, you would have a model for class 1 vs 2, class 1 vs 3, ..., class 1 vs \\(K\\). Then repeat for class 2: 2 vs 3, 2 vs 4, ..., 2 vs \\(|K|\\), and so on.

A third approach is to construct several models using a feature vector dependent on both the data and class label. When given a new input, the model computes

\\[
y = \text{arg}\max\_{y'}\mathbf{w}^T\phi(\mathbf{x},y').
\\]

The margin for this classifier is the distance between the correct class and the closest data point of any other class.


## Additional Resources {#additional-resources}

-   <https://web.mit.edu/6.034/wwwbob/svm.pdf>
-   <https://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf>
