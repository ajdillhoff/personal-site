+++
title = "Stereo Vision"
authors = ["Alex Dillhoff"]
date = 2022-03-23T00:00:00-05:00
tags = ["algorithms", "computer science"]
draft = false
lastmod = 2024-07-10
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Epipolar Geometry](#epipolar-geometry)
- [Calibration with Known Intrinsic Parameters and World Points](#calibration-with-known-intrinsic-parameters-and-world-points)
- [Estimating Depth](#estimating-depth)

</div>
<!--endtoc-->



## Introduction {#introduction}

Binocular vision permits depth perception.
It is an important part of many tasks such as robotic vision, pose estimation, and scene understanding.
The goal of steropsis is to reconstruct a 3D representation of the world given correspondences between two or more cameras.

The process of computing these correspondences assumes two or more cameras with known intrinsic and extrinsic parameters.
Methods exist to estimate the required transformation parameters using points based on matching image features.
If some set of points which a fixed coordinate system is known, such as a calibration pattern, the problem becomes even simpler.
Knowing the exact world point as it is projected in all image planes is essentially a ground truth.

Hartley and Zisserman address three primary questions when dealing with two views:

1.  **Correspondence geometry:** How does a point in one view constraint the corresponding point in a second view?
2.  **Camera geometry:** How do we determine the cameras of both views given a set of corresponding image points?
3.  **Scene geometry:** If we know the cameras and have a set of corresponding points, how can we compute the depth?


## Epipolar Geometry {#epipolar-geometry}

Epipolar geometry is the backbone of stereopsis.
We will first define what epipolar geometry is, how it is used in the stereo vision problem, and the core constraint that limits our search space of point correspondences.
It is defined visually below.

{{< figure src="/ox-hugo/2022-03-24_20-23-56_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Overview of epipolar geometry for stereopsis (Source: Szeliski)." >}}

Consider the point \\(\mathbf{p}\\) whose projection is \\(\mathbf{x}\_0\\) with respect to camera \\(\mathbf{c}\_0\\) and \\(\mathbf{x}\_1\\) with respect to camera \\(\mathbf{c}\_1\\).
All 5 of these points lie on the **epipolar plane**.
Additionally, points \\(\mathbf{e}\_0\\) and \\(\mathbf{e}\_1\\) lie on the line defined by \\(\mathbf{c}\_0\\) and \\(\mathbf{c}\_1\\) as it intersects the image plane of each camera, respectively.
These are called the **epipoles**.
These are also the projections of the other camera centers.

Fundamental to establishing the correspondences between the two cameras is the **epipolar constraint**.
If \\(\mathbf{x}\_0\\) and \\(\mathbf{x}\_1\\) represent projections of the same point, then \\(\mathbf{x}\_1\\) must like on the epipolar line \\(l\_1\\) associated with \\(\mathbf{x}\_0\\) and vice versa.

{{< figure src="/ox-hugo/2022-03-24_21-07-55_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>The epipolar constraint restricts the search space for matching correspondences. (Source: Szeliski)" >}}

As seen in the figure above, potentional matches for \\(\mathbf{x}\_0\\) must lie on the epipolar line defined by \\(\mathbf{e}\_1\\) and \\(\mathbf{x}\_1\\).

Our goal is to compute translation \\(\mathbf{t}\\) and rotation \\(R\\).
To find these, we start by mathematically defining the epipolar constraint.
The epipolar plane is defined by the lines \\(\overrightarrow{\mathbf{c}\_0 \mathbf{p}}\\), \\(\overrightarrow{\mathbf{c}\_1 \mathbf{p}}\\), and \\(\overrightarrow{\mathbf{c}\_0 \mathbf{c}\_1}\\).
This can be written as

\\[
\overrightarrow{\mathbf{c}\_0 \mathbf{p}} \cdot [\overrightarrow{\mathbf{c}\_0 \mathbf{c}\_1} \times \overrightarrow{\mathbf{c}\_1 \mathbf{p}}] = 0.
\\]

We do not know \\(\mathbf{p}\\), but we do have \\(\mathbf{x}\_0\\) and \\(\mathbf{x}\_1\\), the projections of \\(\mathbf{p}\\) onto the image plane of each respective camera.
Assuming that both camera are calibrated, we have a set of known intrinsic matrices \\(\mathbf{K}\_j\\).

Although we do not have the exact \\(z\\) value of the point, we do know the point with respect to the camera calculcated as

\\[
\mathbf{p}\_0 = d\_0 \hat{\mathbf{x}}\_0,
\\]

where

\\[
\hat{\mathbf{x}}\_0 = \mathbf{K}\_0^{-1} \mathbf{x}\_0.
\\]

The relationship between points \\(\mathbf{p}\_0\\) and \\(\mathbf{p}\_1\\) is

\\[
d\_1 \hat{\mathbf{x}}\_1 = R(d\_0 \hat{\mathbf{x}}\_0) + \mathbf{t},
\\]

where \\(R\\) is a rotation matrix and \\(\mathbf{t}\\) is an offset vector.
These are the parameters that are solved from stereo calibration.

Since the vectors \\(\hat{\mathbf{x}}\_0\\), \\(\hat{\mathbf{x}}\_1\\), and \\(\mathbf{t}\\) are coplanar, the plane can be represented by a normal vector.
That is, the vector that is orthogonal to all points in the plane.
Such a vector can be calculated by taking the cross product of both sides of the above equation with \\(\mathbf{t}\\):

\\[
d\_1 [\mathbf{t}]\_{\times} \hat{\mathbf{x}}\_1 = d\_0 [\mathbf{t}]\_{\times} R \hat{\mathbf{x}}\_0,
\\]

where

\\[
[\mathbf{t}]\_{\times}=\begin{bmatrix}
0 & -t\_z & t\_y\\\\
t\_z & 0 & -t\_x\\\\
-t\_y & t\_x & 0
\end{bmatrix}.
\\]

It is true that, since the normal vector is orthogonal to \\(\hat{\mathbf{x}}\_0\\), \\(\hat{\mathbf{x}}\_1\\), and \\(\mathbf{t}\\), taking the dot product of any of these vectors and the normal vector yields 0:

\\[
d\_1 \hat{\mathbf{x}}\_1^T [\mathbf{t}]\_{\times} \hat{\mathbf{x}}\_1 = d\_0 \hat{\mathbf{x}}\_1^T [\mathbf{t}]\_{\times} R \hat{\mathbf{x}}\_0 = 0.
\\]

We have now established the epipolar constraint in terms of the rotation matrix \\(R\\) and translation \\(\mathbf{t}\\).
These are the parameters that relate the points between both cameras.
This constraint is more compactly written as

\\[
\hat{\mathbf{x}}\_1^T E \hat{\mathbf{x}}\_0 = 0,
\\]

where

\\[
E = [\mathbf{t}\_{\times}]R.
\\]

\\(E\\) is called the **essential matrix** which relates the projected points between the two cameras.
Once we have the essential matrix, we can compute \\(\mathbf{t}\\) and \\(R\\).


## Calibration with Known Intrinsic Parameters and World Points {#calibration-with-known-intrinsic-parameters-and-world-points}

We first look at the simplest case of stereo calibration.
The intrinsic parameters of both cameras have been computed using a standard calibration technique.
Additionally, we have a fixed calibration pattern used to establish a correspondence between
the world points and the points with respect to each camera.

Given \\(N\\) measured correspondences \\(\\{(\mathbf{x}\_{i0}, \mathbf{x}\_{i1})\\}\\), we can form a linear system with equations of the form

\begin{alignat\*}{3}
x\_{i0}x\_{i1}e\_{00} & {}+{} & y\_{i0}x\_{i1}e\_{01} & {}+{} & x\_{i1}e\_{02} & {}+{} &\\\\
x\_{i0}y\_{i1}e\_{10} & {}+{} & y\_{i0}y\_{i1}e\_{11} & {}+{} & y\_{i1}e\_{12} & {}+{} &\\\\
x\_{i0}e\_{20} & {}+{} & y\_{i0}e\_{21} & {}+{} & e\_{22} & {}={} & 0
\end{alignat\*}

Given at least 8 equations corresponding to the 8 unknowns of \\(E\\), we can use SVD to solve for \\(E\\).

\\[
E = [\mathbf{t}]\_{\times}R = \mathbf{U \Sigma V^T} = \begin{bmatrix}
\mathbf{u}\_0 & \mathbf{u}\_1 & \mathbf{t}
\end{bmatrix}\begin{bmatrix}
1 & 0 & 0\\\\
0 & 1 & 0\\\\
0 & 0 & 1\\\\
\end{bmatrix}\begin{bmatrix}
\mathbf{v}\_0^T\\\\
\mathbf{v}\_1^T\\\\
\mathbf{v}\_2^T\\\\
\end{bmatrix}
\\]


## Estimating Depth {#estimating-depth}

Given the intrinsic parameters and parameters relating both calibrated cameras, we can estimate the depth of a point that is seen by both cameras.

We know that \\(\mathbf{x}\_0 = K\_0 \hat{\mathbf{x}}\_0\\) and \\(\mathbf{x}\_1 = K\_1 \hat{\mathbf{x}}\_1\\).
With the stereo calibration complete, we also know \\(A = [R\quad \mathbf{t}]\\) and

\\[
\hat{\mathbf{x}}\_0 = A\hat{\mathbf{x}}\_1.
\\]

Plugging into the projection equation for the first camera yields

\\[
\mathbf{x}\_0 = K\_0 A\hat{\mathbf{x}}\_1.
\\]

Our knowns are \\(\mathbf{x}\_0, \mathbf{x}\_1, K\_0, K\_1, \text{ and } A\\).
The only unknown is \\(\hat{\mathbf{x}}\_1\\).

We are left with 2 equations

\begin{align\*}
\mathbf{x}\_0 &= K\_0 A\hat{\mathbf{x}}\_1\\\\
\mathbf{x}\_1 &= K\_1 \hat{\mathbf{x}}\_1.
\end{align\*}

If we let \\(P = K\_0 A\\), we write the \\(\mathbf{x}\_0 = \begin{bmatrix}u\_1\\\v\_1\\\1\end{bmatrix}\\) and

\begin{equation\*}
\begin{bmatrix}
u\_1\\\\
v\_1\\\\
1
\end{bmatrix}=\begin{bmatrix}
p\_{11} & p\_{12} & p\_{13} & p\_{14}\\\\
p\_{21} & p\_{22} & p\_{23} & p\_{24}\\\\
p\_{31} & p\_{32} & p\_{33} & p\_{34}\\\\
\end{bmatrix}\begin{bmatrix}
X\\\\
Y\\\\
Z\\\\
W
\end{bmatrix}.
\end{equation\*}

This gives two equations for \\(x\_1\\) and \\(y\_1\\), the measured 2D feature locations:

\begin{align\*}
x\_1 &= \frac{p\_{11}X + p\_{12}Y + p\_{13}Z + p\_{14}W}{p\_{31}X + p\_{32}Y + p\_{33}Z + p\_{34}W}\\\\
y\_1 &= \frac{p\_{21}X + p\_{22}Y + p\_{23}Z + p\_{24}W}{p\_{31}X + p\_{32}Y + p\_{33}Z + p\_{34}W}.
\end{align\*}

Multiplying both equations by the denominator yields a set of equations that can be solved via linear least squares or SVD.
