+++
title = "Camera Models"
authors = ["Alex Dillhoff"]
date = 2022-03-11T00:00:00-06:00
tags = ["machine learning"]
draft = false
lastmod = 2025-04-01
sections = "Computer Vision"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Reading](#reading)
- [Outline](#outline)
- [Pinhole Model](#pinhole-model)
- [From World Space to Image Space](#from-world-space-to-image-space)
- [Camera Parameters](#camera-parameters)
- [Estimating Camera Parameters](#estimating-camera-parameters)
- [Application: Camera Calibration](#application-camera-calibration)

</div>
<!--endtoc-->



## Reading {#reading}

-   Chapters 1 and 7 (Forsyth and Ponce)
-   <https://www.scratchapixel.com/>
-   <https://docs.google.com/presentation/d/1RMyNQR9jGdJm64FCiuvoyJiL6r3H18jeNIyocIO9sfA/edit#slide=id.p>
-   <http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture8_camera_models_cs131_2016.pdf>
-   <http://vlm1.uta.edu/~athitsos/courses/cse4310_spring2021/lectures/11_geometry.pdf>


## Outline {#outline}

-   Pinhole model
-   Coordinates of a pinhole model
-   Perspective projections
-   Homogeneous coordinates
-   Computer graphics perspective
-   Lenses
-   Intrinsic and extrensic parameters
-   From world to camera to image space
-   Camera calibration


## Pinhole Model {#pinhole-model}

Imagine piercing a small hole into a plate and placing it in front of a black screen.
The light that enters through the pinhole will show an inverted image against the back plane.
If we place a virtual screen in front of the pinhole plate, we can project the image onto it.
This is the basic idea behind a **pinhole camera model**.

{{< figure src="/ox-hugo/2022-03-14_17-02-08_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Pinhole camera model (Forsyth and Ponce)." >}}


### Virtual Camera Coordinates {#virtual-camera-coordinates}

There are two sets of coordinates for each object in the pinhole camera model.
First are the actual coordinates of the visible points as seen from the camera.
For real images, these points are continuous. For virtual images, these are discrete.
To produce an actual image, a set of 3D coordinates are projected onto the image plane.
**These 3D coordinates are with respect to the camera's coordinate system.**

We can represent the transformation from camera coordinates to image coordinates with a projection matrix.
There are many properties of the camera that must be considered, such as the lens, resolution, etc.
We will start with a simple projection matrix.

{{< figure src="/ox-hugo/2022-03-14_19-06-36_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Derivation of simple perspective projection." >}}

In the figure above, the camera is located at point \\(A\\) with its viewing plane defined by \\(BC\\).
The \\(y\\) value of the world space coordinate is given by \\(DE\\).
Notice that there are two right triangles (\\(\Delta ABC\\) and \\(\Delta ADE\\)).
As such, the ratio between their sides is constant:

\\[
\frac{BC}{DE} = \frac{AB}{AD}.
\\]

Solving for \\(y' = BC\\) yields \\(BC = \frac{AB \* DE}{AD}\\).
A similar relationship follows for the \\(x\\) value.
In a more common notation, this is written as

\begin{align\*}
x' &= \frac{x}{z}\\\\
y' &= \frac{y}{z}
\end{align\*}

This can be represented as a transformation matrix and point vector by using **homogeneous coordinates**.
These coordinate are to projective geometry as Cartesian coordinates are to Euclidean geometry.
For a 3D point \\(\mathbf{p} = (X, Y, Z)\\), its corresponding homogeneous coordinate is \\(\mathbf{p}\_w = (x, y, z, w)\\).
The relationship between the original coordinate and its homogeneous coordinate is given by

\\[
X = \frac{x}{w},\quad Y = \frac{y}{w}, \quad Z = \frac{z}{w}.
\\]

With homogeneous coordinates, we can easily project 3D points to an image plane.
First, let the projection matrix be defined as

\\[
M = \begin{bmatrix}
1 & 0 & 0 & 0\\\\
0 & 1 & 0 & 0\\\\
0 & 0 & 1 & 0\\\\
0 & 0 & 1 & 0\\\\
\end{bmatrix}
\\]

and a general homogeneous coordinate is given by

\\[
\mathbf{p} = \begin{bmatrix}
x\\\\
y\\\\
z\\\\
w
\end{bmatrix}.
\\]

Then,

\\[
M\mathbf{p} = \begin{bmatrix}
x\\\\
y\\\\
z\\\\
z
\end{bmatrix}
\\]

and the perspective divide is applied to finish the transformation.
That is, divide each component by \\(w\\) and dropping the last component.

\\[
\mathbf{p}' = \begin{bmatrix}
\frac{x}{z}\\\\
\frac{y}{z}\\\\
1
\end{bmatrix}
\\]


### Physical Camera Coordinates {#physical-camera-coordinates}

If we are simulating a physical camera, then the sensor should be _behind_ the lens.
In that case, the points will be inverted as compared to the virtual model.

{{< figure src="/ox-hugo/2022-03-15_12-10-04_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Visualization of the projection equation (Forsyth and Ponce)." >}}

Remember that this is not taking optics into account.
This is a simple projective model.
The figure below shows the relationship between the virtual image plane and sensor image plane.

{{< figure src="/ox-hugo/2022-03-16_12-36-32_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Sensor plane versus virtual plane." >}}

To calculate the points projected to the sensor plane using our simple matrix from above, we only need to negate the \\(z\\) and \\(w\\) values:

\\[
M = \begin{bmatrix}
1 & 0 & 0 & 0\\\\
0 & 1 & 0 & 0\\\\
0 & 0 & -1 & 0\\\\
0 & 0 & -1 & 0\\\\
\end{bmatrix}
\\]

Then the transformed point becomes

\\[
M\mathbf{p} = \begin{bmatrix}
x\\\\
y\\\\
-z\\\\
-z
\end{bmatrix}.
\\]

Applying the perspective divide yields the 3D point \\((-\frac{x}{z}, -\frac{y}{z}, 1)\\).


#### Considering the Focal Length {#considering-the-focal-length}

An important property that all cameras have is the focal length \\(f\\).
That is the distance between the lens and the camera's sensor.
In computer graphics terms, the focal length may also refer to the distance between the virtual image plane and camera.
This is then called the **near plane** instead.

For the following examples, we will follow OpenGL's projective matrix in which the \\(x\\) and \\(y\\) values are scaled by a relationship between the focal length and image window size:

\\[
\frac{2n}{r - l},
\\]

where \\(n\\) is the near plane distance (related to \\(f\\)), \\(r\\) is the value of the right side of the window, and \\(l\\) is the value of the left side of the window.
For a square window of size \\([-1, 1]\\), this reduces to \\(n\\) (or \\(f\\)).
Then the projection matrix is given as

\\[
M = \begin{bmatrix}
1 & 0 & 0 & 0\\\\
0 & 1 & 0 & 0\\\\
0 & 0 & -\frac{1}{f} & 0\\\\
0 & 0 & -1 & 0\\\\
\end{bmatrix}.
\\]

**Why does \\(M(3, 3) = -\frac{1}{f}\\)?**

The projected point should be \\(f\\) units away from the lens.
Projecting a point using the matrix above yields

\\[
M\mathbf{p} = \begin{bmatrix}
x\\\\
y\\\\
-\frac{z}{f}\\\\
-z
\end{bmatrix}.
\\]

Then applying the perspective divide to compute the projected coordinate results in

\\[
\mathbf{p'} = \begin{bmatrix}
-\frac{fx}{z}\\\\
-\frac{fy}{z}\\\\
\end{bmatrix}.
\\]

We first computed the 3D homogeneous coordinate and then applied the relationship to 2D homogeneous coordinates using the resulting \\((x, y, z)\\) triplet.
Applying the final divide which establishes the relationship between 2D Cartesian and homogeneous coordinates results in \\(\mathbf{p'}\\).

Be aware that computer graphics implementations may apply this process differently using a matrix that looks something like this

\\[
M = \begin{bmatrix}
f & 0 & 0 & 0\\\\
0 & f & 0 & 0\\\\
0 & 0 & -1 & 0\\\\
0 & 0 & -1 & 0\\\\
\end{bmatrix}.
\\]

If we apply this to a point in 3D space using homogeneous coordinates, it results in the point

\\[
M\mathbf{p} = \begin{bmatrix}
fx\\\\
fy\\\\
-z\\\\
-z
\end{bmatrix}.
\\]

Then the resulting 3D point is

\\[
\mathbf{p'} = \begin{bmatrix}
-\frac{fx}{z}\\\\
-\frac{fy}{z}\\\\
1\\\\
\end{bmatrix},
\\]

which gives us the same \\(x\\) and \\(y\\) values of the projected point as before.

{{< figure src="/ox-hugo/2022-03-16_13-44-25_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>3D points (green) projected onto the virtual plane (red) and sensor plane (blue)." >}}


## From World Space to Image Space {#from-world-space-to-image-space}

The process of projecting points in camera space onto an image plane is fairly simple, but gets slightly more complicated
when considering each entity from the _world_ perspective.
Objects in the world appear differently depending on the viewpoint of the camera.
To model this mathematically, we need to be able to describe the points in 3 different spaces:

1.  World space
2.  Camera space
3.  Image space

To fully describe the possibilities that our camera's perspective can take, we need to define scaling, translation, and rotations.
These allow us to describe the different positions and orientations are camera may be in with respect to the world around it.


### Translation {#translation}

Consider the points of a pyramid observed by a virtual camera, as seen below.

{{< figure src="/ox-hugo/2022-03-16_16-11-03_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>Points of a pyramid as seen from our virtual camera." >}}

If we move the camera away from the points, the will appear smaller (assuming a perspective projection).
This movement is described by a translation with respect to the origin in world space.
In the above figure, the camera is position at the origin.
The figure below shows the camera moved down the $z$-axis by 2 units.

{{< figure src="/ox-hugo/2022-03-16_16-14-47_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>Translating the camera backwards by 2 units. The project points appear smaller." >}}

This is described with a matrix as

\\[
T = \begin{bmatrix}
1 & 0 & 0 & -2\\\\
0 & 1 & 0 & 0\\\\
0 & 0 & 1 & 0\\\\
0 & 0 & 0 & 1\\\\
\end{bmatrix}.
\\]

In general, a translation in 3D can be described with the following matrix

\\[
T = \begin{bmatrix}
1 & 0 & 0 & T\_x\\\\
0 & 1 & 0 & T\_y\\\\
0 & 0 & 1 & T\_z\\\\
0 & 0 & 0 & 1\\\\
\end{bmatrix}.
\\]


### Scaling {#scaling}

Projected objects can appear larger or smaller depending on the **focal length** and **field of view**.
Recall the perspective projection matrix from the previous section:

\\[
P = \begin{bmatrix}
f & 0 & 0 & 0\\\\
0 & f & 0 & 0\\\\
0 & 0 & -1 & 0\\\\
0 & 0 & -1 & 0\\\\
\end{bmatrix}.
\\]

The projections we have seen so far are with a focal length of 1.
We can make the objects in the image appear larger by increasing the focal length, as seen below.

{{< figure src="/ox-hugo/2022-03-16_16-23-25_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>Focal length doubled. The projected image appears larger." >}}


### Rotation {#rotation}

Finally, we can rotate our camera about the \\(x\\), \\(y\\), or \\(z\\) axis.
Following a [right-handed coordinate system](https://en.wikipedia.org/wiki/Right-hand_rule), these rotations are given by

\begin{align\*}
R\_x(\theta) &= \begin{bmatrix}
1 & 0 & 0\\\\
0 & \cos \theta & -\sin \theta\\\\
0 & \sin \theta & \cos \theta
\end{bmatrix}\\\\
R\_y(\theta) &= \begin{bmatrix}
\cos \theta & 0 & \sin \theta\\\\
0 & 1 & 0\\\\
-\sin \theta & 0 & \cos \theta
\end{bmatrix}\\\\
R\_z(\theta) &= \begin{bmatrix}
\cos \theta & -\sin \theta & 0\\\\
\sin \theta & \cos \theta & 0\\\\
0 & 0 & 1
\end{bmatrix}\\\\
\end{align\*}

{{< figure src="/ox-hugo/2022-03-16_16-42-17_screenshot.png" caption="<span class=\"figure-number\">Figure 9: </span>Rotating the camera about the y-axis. The points now appear on the right side of the image plane." >}}


## Camera Parameters {#camera-parameters}

When relating points in the world to those of the image space, two sets of parameters are typically used:
the **intrinsic** and **extrinsic** parameters of a camera.
Intrinsic parameters relate the points in the camera's reference frame to those of the image plane.
Extrinsic parameters relate the camera's coordinate system to a world coordinate system as well as describing its position and orientation.


### Intrinsic Parameters {#intrinsic-parameters}

Which parameters are required to project a point to the image plane?
The intrinsic parameters are those which convert the points on the physical sensor into a normalized image frame.
We start with a basic projection matrix

\\[
M = \begin{bmatrix}
1 & 0 & 0 & 0\\\\
0 & 1 & 0 & 0\\\\
0 & 0 & -1 & 0\\\\
0 & 0 & -1 & 0\\\\
\end{bmatrix}.
\\]

A physical sensor may not have square pixels, so there are two additional scale parameters \\(k\\) and \\(l\\).
Along with the focal length \\(f\\), the projected values become

\begin{align\*}
x &= kf\frac{X}{Z} = kf\hat{x}\\\\
y &= lf\frac{Y}{Z} = lf\hat{y}\\\\
\end{align\*}

The focal length and scale factors \\(k\\) and \\(l\\) are dependent and can be written as \\(\alpha = kf\\) and \\(\beta = lf\\).

The physical sensor has an origin at the corner whereas the normalized image frame has an origin at the center.
To account for this offset, a translation parameter for both \\(x\\) and \\(y\\) is necessary.
The pixel conversion is now represented as

\begin{align\*}
x &= \alpha \hat{x} + x\_0\\\\
y &= \beta \hat{y} + y\_0\\\\
\end{align\*}

Finally, the sensor may not be perfectly center.
There may also be additional defects that cause the sensor to be warped or skewed.
This is error that can be accounted for by a rotation.

\begin{align\*}
x &= \alpha \hat{x} - \alpha \cot \theta \hat{y} + x\_0\\\\
y &= \frac{\beta}{\sin \theta} \hat{y} + y\_0\\\\
\end{align\*}

To recap, the 5 intrinsic parameters are

1.  \\(\alpha\\) - scale parameter for \\(x\\)
2.  \\(\frac{\beta}{\sin \theta}\\) - scale parameter for \\(y\\), also accounts for skew
3.  \\(-\alpha \cot \theta\\) - skew coefficient between the \\(x\\) and \\(y\\) axis
4.  \\(x\_0\\) - the principal point for \\(x\\)
5.  \\(y\_0\\) - the principal point for \\(y\\)

The third parameter \\(-\alpha \cot \theta\\) is sometimes represented as \\(\gamma\\) and will be set to 0 in most cases.
A typical intrinsic matrix is then

\\[
K = \begin{bmatrix}
\alpha & \gamma & x\_0\\\\
0 & \beta & y\_0\\\\
0 & 0 & 1\\\\
\end{bmatrix}.
\\]


### Extrinsic Parameters {#extrinsic-parameters}

Given a point with respect to the world frame, \\([\mathbf{p}]\_W\\), we are interested in a change of basis matrix which transforms the world space coordinate to the camera's basis.
This is represented by a change of basis matrix which converts a point in world space to the camera's coordinate frame.

The matrix is defined as a rigid transformation involving a rotation and translation.
The rotation components \\(R\\) are composed following the 3 rotation matrices described earlier.
This only requires 3 parameters if using something like [Euler angles](https://en.wikipedia.org/wiki/Euler_angles).
Euler angles represent the rotational angle around each axis by individual parameters \\(\phi\\), \\(\theta\\), and \\(\psi\\).

\\[
R = \begin{bmatrix}
r\_{11} & r\_{12} & r\_{13}\\\\
r\_{21} & r\_{22} & r\_{23}\\\\
r\_{31} & r\_{32} & r\_{33}\\\\
\end{bmatrix}
\\]

The translational component is represented using an additional 3 parameters \\(t\_x\\), \\(t\_y\\), and \\(t\_z\\).
This can be represented as a matrix

\\[
T = \begin{bmatrix}
1 & 0 & 0 & t\_x\\\\
0 & 1 & 0 & t\_y\\\\
0 & 0 & 1 & t\_z\\\\
\end{bmatrix}.
\\]

For convenience, the extrinsic parameters can be represented as a single matrix

\\[
M = \begin{bmatrix}
r\_{11} & r\_{12} & r\_{13} & t\_x\\\\
r\_{21} & r\_{22} & r\_{23} & t\_y\\\\
r\_{31} & r\_{32} & r\_{33} & t\_z\\\\
\end{bmatrix}
\\]

The full transformation from world space to image space is then represented as

\\[
\mathbf{p} = KM[\mathbf{p}]\_W,
\\]

where \\([\mathbf{p}]\_W = \begin{bmatrix}x\\\y\\\z\\\1\end{bmatrix}\\).


## Estimating Camera Parameters {#estimating-camera-parameters}

There are many applications which require that we know the exact camera parameters, including augmented reality, inpainting techniques, and depth estimation. We now investigate how, given a fixed world coordinate system with known structure, we can approximate the camera parameters.

Consider a set of \\(n\\) image-to-world point correspondences \\(\mathbf{X}\_i \leftrightarrow \mathbf{x}\_i\\). The goal is to compute a homography \\(\mathbf{H}\\) that relates the image points to the world points: \\(\mathbf{x}\_i = \mathbf{H}\mathbf{X}\_i\\). This relationship can be written as \\(\mathbf{x}\_i \times \mathbf{H}\mathbf{X}\_i = 0\\) because the cross product of two vectors is zero if they are parallel. \\(H\mathbf{x}\_i\\) can be written as

\begin{pmatrix}
\mathbf{h}\_1^T\mathbf{x}\_i\\\\
\mathbf{h}\_2^T\mathbf{x}\_i\\\\
\mathbf{h}\_3^T\mathbf{x}\_i\\\\
\end{pmatrix}

Then the cross product is given by

\begin{pmatrix}
y\_i\mathbf{h}\_3^T\mathbf{X}\_i - w\_i\mathbf{h}\_2^T\mathbf{X}\_i\\\\
w\_i\mathbf{h}\_1^T\mathbf{X}\_i - x\_i\mathbf{h}\_3^T\mathbf{X}\_i\\\\
x\_i\mathbf{h}\_2^T\mathbf{X}\_i - y\_i\mathbf{h}\_1^T\mathbf{X}\_i\\\\
\end{pmatrix}

Observing that \\(\mathbf{h}\_j^T\mathbf{X}\_i = \mathbf{X}\_i^T\mathbf{h}\_j\\), we can write this as a homogeneous linear system.

\\[
\begin{bmatrix}
\mathbf{0}^T & -w\_i\mathbf{X}\_i^T & y\_i\mathbf{X}\_i^T\\\\
w\_i\mathbf{X}\_i^T & \mathbf{0}^T & -x\_i\mathbf{X}\_i^T\\\\
-y\_i\mathbf{X}\_i^T & x\_i\mathbf{X}\_i^T & \mathbf{0}^T\\\\
\end{bmatrix} \begin{pmatrix}
\mathbf{h}\_1\\\\
\mathbf{h}\_2\\\\
\mathbf{h}\_3\\\\
\end{pmatrix} = \mathbf{0}
\\]

This is a linear system whose third equation is linearly dependent on the first two, thus we only need to solve the first two equations.

\\[
\begin{bmatrix}
\mathbf{0}^T & -w\_i\mathbf{X}\_i^T & y\_i\mathbf{X}\_i^T\\\\
w\_i\mathbf{X}\_i^T & \mathbf{0}^T & -x\_i\mathbf{X}\_i^T\\\\
\end{bmatrix} \begin{pmatrix}
\mathbf{h}\_1\\\\
\mathbf{h}\_2\\\\
\mathbf{h}\_3\\\\
\end{pmatrix} = \mathbf{0}
\\]

In practice, this system is denoted as \\(A\_i \mathbf{h} = \mathbf{0}\\) where \\(A\_i\\) is a \\(2 \times 9\\) matrix and \\(\mathbf{h}\\) is a \\(9 \times 1\\) vector. This implies that we need at least 4 point correspondences to solve for \\(\mathbf{h}\\). These correspondences can be used to create 4 matrices \\(A\_i\\) which are then stacked to form a \\(8 \times 9\\) matrix \\(A\\). **But wait... \\(A\\) is only rank 8!** This is because the solution can only be determined up to a scale factor.

If we the matrix had rank 9, then the solution would be trivial. Remember, we're working with a homogeneous linear system.


### Relationship to Camera Parameters {#relationship-to-camera-parameters}

Given the basic solution, let's relate this to the camera parameters. Our goal is to estimate \\(P\\) which is a \\(3 \times 4\\) matrix. This matrix is composed of the intrinsic matrix \\(K\\) and the extrinsic matrix \\(M\\). Since each row of \\(P\\) is a 4-vector, each \\(A\_i\\) is a \\(2 \times 12\\) matrix.


### Algebraic versus Geometric Error {#algebraic-versus-geometric-error}

The solution to the linear system minimizes the algebraic error. Even if the points are perfectly matched, the solution may not be accurate. This is because the solution is only determined up to a scale factor. The geometric error is minimized by using a non-linear optimization technique such as [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm).


## Application: Camera Calibration {#application-camera-calibration}

Popular software solutions for calibrating cameras are provided by [MATLAB](http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example.html) and [OpenCV](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html). These are based off of work by [Zhengyou Zhang](https://www.microsoft.com/en-us/research/project/a-flexible-new-technique-for-camera-calibration-2/?from=https%3A%2F%2Fresearch.microsoft.com%2F%7Ezhang%2FCalib%2F). We will follow the original publication to explain the solutions to camera calibration. As such, the notation will change a bit.

The first goal is to relate the model points \\(\mathbf{M}\\) to the image points \\(\mathbf{m}\\) by a homography \\(\mathbf{H}\\) such that

\\[
s\hat{\mathbf{m}} = \mathbf{H} \hat{\mathbf{M}},
\\]

where \\(\hat{\mathbf{m}}\\) is the homogeneous coordinate of \\(\mathbf{m}\\) (likewise for \\(\mathbf{M}\\)) and \\(s\\) is a scale factor.

To relate this to the notation in the previous section,

\\[
s\tilde{\mathbf{m}} = \mathbf{H} \tilde{\mathbf{M}} \equiv \mathbf{p} = KM[\mathbf{p}]\_W.
\\]

In the paper, \\(\mathbf{H} = \mathbf{A}[\mathbf{R}\quad \mathbf{t}]\\), where

\\[
\mathbf{A} = \begin{bmatrix}
\alpha & \gamma & u\_0\\\\
0 & \beta & v\_0\\\\
0 & 0 & 1\\\\
\end{bmatrix}.
\\]

\\([\mathbf{R}\quad \mathbf{t}]\\) is the matrix \\(M\\) from the previous section.


### Estimating the Homography {#estimating-the-homography}

If we have an image of the model plane, we can estimate \\(\mathbf{H}\\) using maximum-likelihood estimation as follows.
Assume that \\(\mathbf{m}\_i\\) is affected by Gaussian noise with zero mean and covariance \\(\mathbf{\Lambda}\_{\mathbf{m}\_i}\\).
The objective is then to minimize

\\[
\sum\_i (\mathbf{m}\_i - \hat{\mathbf{m}\_i})^T \mathbf{\Lambda\_{\mathbf{m}\_i}^{-1}}(\mathbf{m}\_i - \hat{\mathbf{m}\_i}),
\\]

where

\\[
\hat{\mathbf{m}\_i} = \frac{1}{\bar{\mathbf{h}}\_3^T \mathbf{M}\_i} \begin{bmatrix}
\bar{\mathbf{h}}\_1^T \mathbf{M}\_i\\\\
\bar{\mathbf{h}}\_2^T \mathbf{M}\_i\\\\
\end{bmatrix}.
\\]

\\(\bar{\mathbf{h}}\_i\\) represents the $i$th row of \\(\mathbf{H}\\).

The approach now constructs a matrix of model points similar to the approach used by [RANSAC]({{< relref "random_sample_consensus.md" >}}).
First, let \\(\mathbf{x} = [\bar{\mathbf{h}}\_1^T\  \bar{\mathbf{h}}\_2^T\ \bar{\mathbf{h}}\_3^T]\\).
Recalling that \\(\widetilde{\mathbf{M}}^T \in [X, Y, 1]^T\\),

\\[
\begin{bmatrix}
\widetilde{\mathbf{M}}^T & \mathbf{0}^T & -u \widetilde{\mathbf{M}}^T\\\\
\mathbf{0}^T & \widetilde{\mathbf{M}}^T & -v \widetilde{\mathbf{M}}^T\\\\
\end{bmatrix}\mathbf{x} = \mathbf{0}.
\\]

Writing this as \\(\mathbf{L}\mathbf{x} = \mathbf{0}\\), where \\(\mathbf{L} \in \mathbb{R}^{2n \times 9}\\) with \\(n\\) being the number of points, the solution the eigenvector of \\(\mathbf{L}^T\mathbf{L}\\) corresponding to the smallest eigenvalue.
This can be computed using [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) without computing \\(\mathbf{L}^T\mathbf{L}\\) directly.
Factorize \\(\mathbf{L} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T\\) using SVD.
Then the parameters that minimizes the objective is the column vector in \\(\mathbf{V}\\) corresponding to the smallest eigenvalue.

It is noted that since the columns of the rotation matrix are orthonormal, the following constraints are observed:

\begin{align\*}
\mathbf{h}\_1^T \mathbf{A}^{-T} \mathbf{A}^{-1} \mathbf{h}\_2 &= 0\\\\
\mathbf{h}\_1^T \mathbf{A}^{-T} \mathbf{A}^{-1} \mathbf{h}\_1 &= \mathbf{h}\_2^T \mathbf{A}^{-T} \mathbf{A}^{-1} \mathbf{h}\_2\\\\
\end{align\*}


### Solving for the Camera Parameters {#solving-for-the-camera-parameters}

With the homography \\(\mathbf{H}\\) and constraints given in the last section, Zhang proposes a two step solution that first estimates the parameters using a closed form solution.
These serve as a starting point for a non-linear optimization using [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm).

First, let

\\[
\mathbf{B} = \mathbf{A}^{-T} \mathbf{A}^{-1} = \begin{bmatrix}
\frac{1}{\alpha^2} & -\frac{\gamma}{\alpha^2\beta} & \frac{v\_0\gamma - u\_0\beta}{\alpha^2 \beta}\\\\
-\frac{\gamma}{\alpha^2\beta} & \frac{\gamma^2}{\alpha^2\beta^2} + \frac{1}{\beta^2} & -\frac{\gamma(v\_0\gamma-u\_0\beta)}{\alpha^2\beta^2}-\frac{v\_0}{\beta^2}\\\\
\frac{v\_0\gamma - u\_0\beta}{\alpha^2 \beta} & -\frac{\gamma(v\_0\gamma-u\_0\beta)}{\alpha^2\beta^2}-\frac{v\_0}{\beta^2} & \frac{(v\_0\gamma-u\_0\beta)^2}{\alpha^2\beta^2}+\frac{v\_0^2}{\beta^2}+1
\end{bmatrix}.
\\]

This is a symmetric matrix which can be represented more compactly as

\\[
\mathbf{b} = [B\_{11}\ B\_{12}\ B\_{22}\ B\_{13}\ B\_{23}\ B\_{33}]^T.
\\]

We can then write \\(\mathbf{h}\_1^T \mathbf{A}^{-T} \mathbf{A}^{-1} \mathbf{h}\_2 = 0\\) as

\\[
\mathbf{h}\_i^T \mathbf{B} \mathbf{h}\_j = \mathbf{v}\_{ij}^T \mathbf{b},
\\]

where

\\[
\mathbf{v}\_{ij} = [h\_{i1}h\_{j1}, h\_{i1}h\_{j2} + h\_{i2}h\_{j1}, h\_{i2}h\_{j2}, h\_{i3}h\_{j1} + h\_{i1}h\_{j3}, h\_{i3}h\_{j2}+h\_{i2}h\_{j3}, h\_{i3}h\_{j3}]^T.
\\]

This permits us to write the constraints from the previous section as two homogeneous equations in \\(\mathbf{b}\\)

\\[
\begin{bmatrix}
\mathbf{v}\_{12}^T\\\\
(\mathbf{v}\_{11} - \mathbf{v}\_{22})^T
\end{bmatrix} \mathbf{b} = \mathbf{0}.
\\]

If we compute the homography, as above, on \\(n\\) images, we end up with \\(n\\) equations like the one directly above.
Stacking this yields

\\[
\mathbf{V}\mathbf{b} = \mathbf{0},
\\]

where \\(\mathbf{V} \in \mathbb{R}^{2n \times 6}\\).

The initial set of parameters is then solved following the solution for each homography \\(\mathbf{H}\\).
With \\(\mathbf{b}\\) estimated, the intrinsic parameters are

\begin{align\*}
v\_0 &= (B\_{12}B\_{13} - B\_{11}B\_{23})/(B\_{11}B\_{22} - B\_{12}^2)\\\\
\lambda &= B\_{33} - [B\_{13}^2 + v\_0(B\_{12}B\_{13} - B\_{11}B\_{23})]/B\_{11}\\\\
\alpha &= \sqrt{\lambda / B\_{11}}\\\\
\beta &= \sqrt{\lambda B\_{11}/(B\_{11}B\_{22} - B\_{12}^2)}\\\\
\gamma &= -B\_{12}\alpha^2\beta / \lambda\\\\
u\_0 &= \gamma v\_0 / \alpha - B\_{13} \alpha^2 / \lambda.
\end{align\*}

The parameters \\(\alpha, \beta, \gamma, u\_0, v\_0\\) make up our intrinsic parameter matrix \\(\mathbf{A}\\).
These can be used to compute the extrinsic parameters following

\begin{align\*}
\lambda &= 1 / \\|\mathbf{A}^{-1}\mathbf{h}\_1\\|\\\\
\mathbf{r}\_1 &= \lambda \mathbf{A}^{-1}\mathbf{h}\_1\\\\
\mathbf{r}\_2 &= \lambda \mathbf{A}^{-1}\mathbf{h}\_2\\\\
\mathbf{r}\_3 &= \mathbf{r}\_1 \times \mathbf{r}\_2\\\\
\mathbf{t} &= \lambda \mathbf{A}^{-1}\mathbf{h}\_3\\\\
\end{align\*}


### Refining the Estimates {#refining-the-estimates}

The approach in the last section is produced by minimizing algebraic distances.
These values are further refined using maximum-likelihood estimation by way of a nonlinear optimizer.
If there are \\(m\\) points on each of the \\(n\\) images, the objective function to minimize is given by

\\[
\sum\_{i=1}^n \sum\_{j=1}^m \\|\mathbf{m}\_{ij} - \hat{\mathbf{m}}(\mathbf{A}, \mathbf{R}\_i, \mathbf{t}\_i, \mathbf{M}\_j)\\|^2,
\\]

where \\(\hat{\mathbf{m}}(\mathbf{A}, \mathbf{R}\_i, \mathbf{t}\_i, \mathbf{M}\_j)\\) is a projection of point \\(\mathbf{M}\_j\\) in image \\(i\\).
This function is minimized using the [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) algorithm.
