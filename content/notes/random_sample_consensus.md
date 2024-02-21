+++
title = "RANdom SAmple Consensus"
authors = ["Alex Dillhoff"]
date = 2022-02-15T00:00:00-06:00
tags = ["machine learning"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Finding the Best Fit Model](#finding-the-best-fit-model)

</div>
<!--endtoc-->



## Introduction {#introduction}

Unless our data is perfect, we will not be able to find parameters that fit the data in the presence of outliers.
Consider fitting the data in the figure below using a least squares method.

{{< figure src="/ox-hugo/2024-02-20_19-46-07_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Points sample along a line with many outliers around it. Source: Wikipedia" >}}

If we were to fit a naive least squares model, the outliers would surely produce parameters for a line that does not fit the most amount of data possible.

Consider the figures below. In the first one, a least squares model is fit to points generated from a line.
With the addition of just a single outlier, the model no longer fits the line.

{{< figure src="/ox-hugo/2024-02-20_19-46-28_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Least squares can easily fit a line with great accuracy." >}}

{{< figure src="/ox-hugo/2024-02-20_19-46-52_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>A single outlier leads to a bad fit for linear regression." >}}

Ideally, we want a model that is robust to outliers.
That is, the model should be fit such that it matches the largest number of samples, or **inliers**.
One such approach to this problem is **RANdom SAmple Consensus (RANSAC)**.

{{< figure src="/ox-hugo/2024-02-20_19-47-10_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>RANSAC fit to most inliers while ignoring the outliers. Source: Wikipedia" >}}

The general process is as follows:

1.  Randomly select source samples and their matching targets.
2.  Fit a model to the data such that transforming the input by the model parameters yields a close approximation to the targets.
3.  Measure the error of how well ALL data fits and select the number of inliers with error less than \\(t\\).
4.  If the error is lower than the previous best error, fit a new model to these inliers.

{{< figure src="/ox-hugo/2024-02-20_19-47-30_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>RANSAC fitting random samples and counting the number of inliers. Source: Wikipedia" >}}

The algorithm can be found on the [Wikipedia page](https://en.wikipedia.org/wiki/Random_sample_consensus).


## Finding the Best Fit Model {#finding-the-best-fit-model}

When it comes to finding the parameters of a transformation matrix that converts points in one image to another, how do we solve for that matrix? We are looking for some \\(A\\) such that

\begin{equation\*}
A\mathbf{x} = \mathbf{\hat{x}}.
\end{equation\*}

In a perfect world, \\(\mathbf{\hat{x}}\\) will match the target point \\(\mathbf{y}\\). In other words,

\\(\\|\mathbf{\hat{x}} - \mathbf{y}\\|\_{2} = 0\\).

For an affine transformation, we would have some transformation matrix

\begin{equation\*}
A =
\begin{bmatrix}
a\_{11} & a\_{12} & a\_{13}\\\\
a\_{21} & a\_{22} & a\_{23}
\end{bmatrix}.
\end{equation\*}

Then we compute each component of \\(A\mathbf{x}\\) as

\begin{align\*}
\hat{x}\_1 &= a\_{11} \* x\_1 + a\_{12} \* x\_2 + a\_{13} \* 1\\\\
\hat{x}\_2 &= a\_{21} \* x\_1 + a\_{22} \* x\_2 + a\_{23} \* 1\\\\
\end{align\*}

We can fit this using a least squares approach by the following construction.

\begin{equation\*}
\begin{bmatrix}
x\_1^{(1)} & x\_2^{(1)} & 1 & 0 & 0 & 0\\\\
0 & 0 & 0 & x\_1^{(1)} & x\_2^{(1)} & 1\\\\
&& \vdots\\\\
x\_1^{(n)} & x\_2^{(n)} & 1 & 0 & 0 & 0\\\\
0 & 0 & 0 & x\_1^{(n)} & x\_2^{(n)} & 1\\\\
\end{bmatrix}
\begin{bmatrix}
a\_{11}\\\\
a\_{12}\\\\
a\_{13}\\\\
a\_{21}\\\\
a\_{22}\\\\
a\_{23}\\\\
\end{bmatrix}=
\begin{bmatrix}
\hat{x}\_1^{(1)}\\\\
\hat{x}\_2^{(1)}\\\\
\vdots\\\\
\hat{x}\_1^{(n)}\\\\
\hat{x}\_2^{(n)}\\\\
\end{bmatrix}
\end{equation\*}

We can solve this analytically! Recall the **normal equations**:

\\[
A^T A \mathbf{x} = A^T \mathbf{b}.
\\]

Let's test this on a couple of images...

{{< figure src="/ox-hugo/2022-02-15_19-20-07_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>Two images taken with matching features shared between them." >}}

First, we use some feature detector such as [SIFT]({{< relref "scale_invariant_feature_transforms.md" >}}) to find keypoints in each image.
Then, we can take a brute force approach to determine which keypoints match between them.

{{< figure src="/ox-hugo/2022-02-15_19-24-24_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>We've got a lot of potential matches here." >}}

After running RANSAC, we end up with a model that fits the following inlier points.

{{< figure src="/ox-hugo/2022-02-15_19-25-31_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>Many of the outliers were removed and we are left with the following matches." >}}

We can use the found transformation matrix to warp our source image to fit our destination image as seen below.

{{< figure src="/ox-hugo/2022-02-15_19-31-23_screenshot.png" caption="<span class=\"figure-number\">Figure 9: </span>Images stitched together... not perfect!" >}}


### Finding a better transformation {#finding-a-better-transformation}

The transformation matrix was an affine transformation matrix.
What we really want is a projective transformation!
We can extend our approach to finding an affine matrix from earlier by remembering that projective transformations are completed following a perspective divide (usually denoted by \\(w\\)).

Instead of a constant 1 in the third position of the affine vector, we have a value \\(w\\):

\begin{equation\*}
\begin{bmatrix}
a\_{11} & a\_{12} & a\_{13}\\\\
a\_{21} & a\_{22} & a\_{23}\\\\
a\_{31} & a\_{32} & a\_{33}\\\\
\end{bmatrix}
\begin{bmatrix}
x\\\\
y\\\\
w
\end{bmatrix}=
\begin{bmatrix}
\hat{x}\\\\
\hat{y}\\\\
\hat{w}
\end{bmatrix}.
\end{equation\*}

Dividing by \\(\hat{w}\\) completes the perspective projection:

\begin{equation\*}
\begin{bmatrix}
\frac{\hat{x}}{\hat{w}}\\\\
\frac{\hat{y}}{\hat{w}}\\\\
1
\end{bmatrix}.
\end{equation\*}

Again, we can write out the individual equation for each component as

\begin{align\*}
\hat{x} &= (a\_{11} \* x + a\_{12} \* y + a\_{13} \* w) \div (a\_{31} \* x + a\_{32} \* y + a\_{33} \* w)\\\\
\hat{y} &= (a\_{21} \* x + a\_{22} \* y + a\_{23} \* w) \div (a\_{31} \* x + a\_{32} \* y + a\_{33} \* w)\\\\
\end{align\*}

We may assume that \\(w = 1\\) for the original points (before transformation).
Additionally, \\(a\_{33}\\) is typically set to 1 when constructing a transformation matrix.
These are safe enough assumptions to make considering that we will make many attempts at finding the best fitting parameters.

Solving for \\(\hat{x}\\) and \\(\hat{y}\\) in terms of a linear combination of elements yields

\begin{align\*}
\hat{x} &= a\_{11} \* x + a\_{12} \* y + a\_{13} - \hat{x} \* a\_{31} \* x - \hat{x} \* a\_{32} \* y\\\\
\hat{y} &= a\_{21} \* x + a\_{22} \* y + a\_{23} - \hat{y} \* a\_{31} \* x - \hat{y} \* a\_{32} \* y\\\\
\end{align\*}

We can fit this using a least squares approach by the following construction.

\begin{equation\*}
\begin{bmatrix}
x\_1^{(1)} & x\_2^{(1)} & 1 & 0 & 0 & 0 & -x\_1^{(1)}\hat{x}\_1^{(1)} & -x\_2^{(1)}\hat{x}\_1^{(1)}\\\\
0 & 0 & 0 & x\_1^{(1)} & x\_2^{(1)} & 1 & -x\_1^{(1)}\hat{x}\_2^{(1)} & -x\_2^{(1)}\hat{x}\_2^{(1)}\\\\
&& \vdots\\\\
x\_1^{(n)} & x\_2^{(n)} & 1 & 0 & 0 & 0 & -x\_1^{(n)}\hat{x}\_1^{(n)} & -x\_2^{(n)}\hat{x}\_1^{(n)}\\\\
0 & 0 & 0 & x\_1^{(n)} & x\_2^{(n)} & 1 & -x\_1^{(n)}\hat{x}\_2^{(n)} & -x\_2^{(n)}\hat{x}\_2^{(n)}
\end{bmatrix}
\begin{bmatrix}
a\_{11}\\\\
a\_{12}\\\\
a\_{13}\\\\
a\_{21}\\\\
a\_{22}\\\\
a\_{23}\\\\
a\_{31}\\\\
a\_{32}
\end{bmatrix}=
\begin{bmatrix}
\hat{x}\_1^{(1)}\\\\
\hat{x}\_2^{(1)}\\\\
\vdots\\\\
\hat{x}\_1^{(n)}\\\\
\hat{x}\_2^{(n)}
\end{bmatrix}.
\end{equation\*}

We can use the normal equations as before to solve for this system.

The figure below shows the final result of image stitching using a perspective projection instead of an affine matrix.

{{< figure src="/ox-hugo/2022-02-16_17-59-40_screenshot.png" caption="<span class=\"figure-number\">Figure 10: </span>Stitching using a perspective projection." >}}
