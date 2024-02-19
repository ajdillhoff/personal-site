+++
title = "Optical Flow"
authors = ["Alex Dillhoff"]
date = 2022-03-06T00:00:00-06:00
tags = ["computer vision"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Motion Features](#motion-features)
- [Computing Optical Flow](#computing-optical-flow)
- [Assumptions of Small Motion](#assumptions-of-small-motion)

</div>
<!--endtoc-->



## Introduction {#introduction}

Optical flow refers to the apparent motion in a 2D image.
A motion field refers to the true motion of objects in 3D.
The idea of optical flow features is a simple concept.
If a fixed camera records a video of someone walking from the left side of the screen to the right, a difference of two consecutive frames reveals much about the apparent motion.

**TODO** Frame differencing figure

Consider a sphere with Lambertian reflectance.

If the sphere is rotated:

1.  What does the motion field look like?
2.  What does the optical flow look like?

If, instead, a point light is rotated around the sphere:

1.  What does the motion field look like?
2.  What does the optical flow look like?


## Motion Features {#motion-features}

Motion features are used a wide variety of tasks from compression, image segmentation, tracking, detection, video de-noising, and more.
Consider a fixed camera with a sphere centered in the middle of the frame.
As the frame moves towards the camera, its apparent size will become larger.
If we were to analyze the optical flow of such a sequence, we would see that the flow is radial as the sphere, projected as a circle, grows.

{{< figure src="/ox-hugo/2022-03-06_17-16-16_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>From \"Computer Vision - A Modern Approach\" by Forsyth and Ponce" >}}

At \\(t=1\\), the radius of the circle is given as \\(R\\).
At \\(t=2\\), the radius is \\(r = f\frac{R}{Z}\\), where \\(f\\) is some function of the motion and \\(Z\\) is the distance between the sphere and the camera.
From this, we can also compute the speed at which the sphere is travelling towards the camera as \\(V=\frac{dZ}{dt}\\).
The apparent rate of growth as observed by the camera is \\(\frac{dr}{dt} = -f\frac{RV}{Z^2}\\).
We can also determine the time to contact with the camera as \\(-\frac{Z}{V}\\).

{{< figure src="/ox-hugo/2022-03-06_18-06-27_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Optical flow on different parts of the image as observed from a moving camera whose direction of focus is perpendicular to the white plane. Source: \"Computer Vision - A Modern Approach\" by Forsyth and Ponce" >}}

In the figure above, the observer is moving at a constant rate to the left.
The points in the image appear to translate towards the right edge of the frame.
Points that are close to the camera appear to move faster than those that are farther away.
This can be used to estimate the apparent depth between objects in a scene.


## Computing Optical Flow {#computing-optical-flow}

A popular assumption for optical flow, as discussed in <&hornDeterminingOpticalFlow1980>, is that of brightness constancy.
That is, a local feature has the same image intensity in one frame as it does in the subsequent frame.

\\[
I(x + u, y + v, t + 1) = I(x, y, t)
\\]

To formulate the problem as a differentiable function, an additional spatial smoothness assumption is made.
It is assumed that a pixel's neighbors will also have the same optical flow as the pixel in question.
This is a reasonable assumption that asserts that patches are locally rigid and will exhibit uniform motion.

Given these two constraints, we can formulate an objective function.
First, the objective function assuming brightness constancy is given by

\\[
E\_D(\mathbf{u}, \mathbf{v}) = \sum\_{s}(I(x\_s + u\_s, y\_s + v\_s, t + 1) - I(x, y, t))^2.
\\]

Adding in the assumption of uniform motion in a local region yields the term

\\[
E\_S(\mathbf{u}, \mathbf{v}) = \sum\_{n\in G(s)}(u\_s - u\_n)^2 + \sum\_{n \in G(s)}(v\_s - v\_n)^2.
\\]

Putting these together, with a weighting term, yields

\\[
E(\mathbf{u}, \mathbf{v}) = E\_D + \lambda E\_S.
\\]

This energy function \\(E\_D\\) is minimized by differentiating and setting the equation to 0:

\\[
I\_x u + I\_y v + I\_t = 0.
\\]

The entire energy to be minimized is then

\\[
E\_D(\mathbf{u}, \mathbf{v}) = \sum\_{s}(I\_{x,s}u\_s + I\_{y,s}v\_s + I\_{t,s})^2 + \lambda \sum\_{n\in G(s)}(u\_s - u\_n)^2 + \sum\_{n \in G(s)}(v\_s - v\_n)^2.
\\]

Differentiating this and setting to 0 yields two equations in \\(u\\) and \\(v\\):

\begin{align\*}
\sum\_s{(I\_{x,s}^2 u\_s + I\_{x,s}I\_{y,s}v\_s + I\_{x,s}I\_{t,s}) + \lambda \sum\_{n \in G(s)}(u\_s - u\_n)} &= 0\\\\
\sum\_s{(I\_{x,s}I\_{y,s} u\_s + I\_{y,s}^2v\_s + I\_{y,s}I\_{t,s}) + \lambda \sum\_{n \in G(s)}(v\_s - v\_n)} &= 0\\\\
\end{align\*}

Note that this is computed for every pixel in the image.
This system is no longer underspecified because of the assumption that neighbors will exhibit the same flow.
We now have 5 equations per pixel.
In more recent works, larger neighborhood grids (\\(5 \times 5\\)) are used.
Then, we have 25 equations per pixel.
Since this is a system of linear equations, it could be computed directly using the normal equations.

However, Horn and Schunck did not have very fast computers in 1981.
So, they introduced an iterative solution <&hornDeterminingOpticalFlow1980>.


## Assumptions of Small Motion {#assumptions-of-small-motion}

One of the core assumptions in early formulations of optical flow is that motion is very small (&lt;1 pixel).
In reality, some objects may move over 100 pixels within a single frame.
A simple solution to this problem was proposed by Bergen et al. in 1992 <&bergenHierarchicalModelbasedMotion>.
By creating an image pyramid over several resolutions, the assumption of small motion at each scale is still reasonable.

{{< figure src="/ox-hugo/2022-03-06_19-37-48_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Hierarchical motion estimation (Bergen et al.)" >}}

At one scale, the warping parameters are estimated.
Next, they are used to warp the image to match the one at \\(t-1\\).
The warped image and true image at \\(t-1\\) are compared to refine the parameters.
The refined parameters are then sent to the next scale layer.
