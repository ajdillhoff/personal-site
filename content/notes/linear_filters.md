+++
title = "Linear Filters"
authors = ["Alex Dillhoff"]
date = 2022-01-22T00:00:00-06:00
tags = ["computer vision"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Smoothing](#smoothing)
- [Convolution](#convolution)
- [Gaussian Filters](#gaussian-filters)
- [Image Derivatives](#image-derivatives)

</div>
<!--endtoc-->



## Introduction {#introduction}

-   How do we detect specific patterns in images (eyes, nose, spots, etc.)?
-   Weighted sums of pixel values.


## Smoothing {#smoothing}

When discussing resizing and interpolation, we saw how the choice of scale factor and rotation can produce aliasing in images. Typically, this effect is hidden using some sort of smoothing.

Let's first look at the case of downsampling an image to 10\\% of its original size. If we use nearest neighbor interpolation, the result is very blocky, as seen below.

{{< figure src="/ox-hugo/2022-01-30_20-49-14_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Rocinante cropped and scaled to 10\\% of the original size. Source: The Expanse" >}}

Effectively, an entire block of pixels in the original image is being mapped to the nearest neighbor. We are losing a lot of information from the original image. The figure below shows an overlay of the low resolution grid over a patch of the high resolution image.

{{< figure src="/ox-hugo/2022-01-30_21-48-58_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>The center dots show the selected pixel value for the downsampled grid." >}}


### Computing the Average {#computing-the-average}

Instead of naively selecting one pixel to represent an entire block, we could compute the average pixel value of all pixels within that block. This is a simple as taking an equal contribution from each pixel in the block and dividing by the total number of pixels in the block. An example of such a block is shown below.

{{< figure src="/ox-hugo/2022-01-30_21-55-06_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>A \\(3 \times 3\\) averaging filter." >}}

By varying the size of this filter, we can effectively change the factor for which we smooth the aliasing.

{{< figure src="/ox-hugo/2022-01-30_21-32-36_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Original image smoothed with \\(9 \times 9\\) average filter before downsampling to 10\\% of its original size." >}}


## Convolution {#convolution}

How can we apply the averaging filter to an image effectively? We slide the filter across the image starting at the first pixel in the first row and move row by row until all pixels have been computed.

{{< figure src="/ox-hugo/2022-01-30_22-20-56_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>The takes input from all pixels under it and computes the average." >}}

{{< figure src="/ox-hugo/2022-01-30_23-13-19_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>Final portion of image after kernel is applied." >}}

This is performed by the **convolution** operation. It is defined as

\begin{equation\*}
g(x, y) = \omega \* f(x, y) = \sum\_{dx = -a}^a \sum\_{dy=-b}^b \omega(dw, dy)f(x + dx, y + dy).
\end{equation\*}

The range \\((-a, a)\\) represents the rows of the kernel and \\((-b, b)\\) the range of the columns of the kernel. The center of the kernel is at \\((dx = 0, dy = 0)\\).

This operation is one of, if not the most, important operations in computer vision. It is used to apply filters, but we will later see it as one of the guiding operators for feature extraction and deep learning methods.

{{< figure src="/ox-hugo/2022-01-31_15-40-19_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>An image \\(f\\) convolved with kernel \\(h\\). Source: Szeliski" >}}


### Properties {#properties}


#### Commutativity {#commutativity}

\\(f \* g = g \* f\\)


#### Associativity {#associativity}

\\(f \* (g \* h) = (f \* g) \* h\\)


#### Distributivity {#distributivity}

\\(f \* (g + h) = (f \* g) + (f \* h)\\)


### Shift Invariant Linear Systems {#shift-invariant-linear-systems}

Convolution is a _linear shift-invariant_ operator. That is, it obeys the following properties.

**Superposition:** The response of the sum of the input is the sum of the individual responses.

\\[
R(f + g) = R(f) + R(g)
\\]

**Scaling:** The response to a scaled input is equal to the scaled response of the same input.

\\[
R(kf) = kR(f)
\\]

**Shift Invariance:** The response to a translated input is equal to the translation of the response to the input.

A **system** is linear if it satisfies both the superposition and scaling properties. Further, it is a shift-invariant linear system if it is linear and satisfies the shift-invariance property.

**Is the average box filter linear?** Yes, it is applied with convolution which behaves the same everywhere.

**Is thresholding a linear system?** No, it can be shown that \\(f(n, m) + g(n, m) > T\\), but \\(f(n, m) < T\\) and \\(g(n, m) < T\\).


## Gaussian Filters {#gaussian-filters}

Blurring an image using a box filter does not simulate realistic blurring as well as Gaussian filters do. The following figure exemplifies the difference between the two.

{{< figure src="/ox-hugo/2022-01-31_16-57-33_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>The left image is blurred using a uniform average box filter. The right image is blurred with a Gaussian filter." >}}

The shortcomings of the average box filter can be seen when viewing the artifacting visible at edges, especially at corners.

A Guassian kernel is defined as

\begin{equation\*}
G(x, y;\sigma) = \frac{1}{2\pi \sigma^2}\exp\Big(-\frac{(x^2 + y^2)}{2 \sigma^2}\Big).
\end{equation\*}

In effect, it enforces a greater contribution from neighbors near the pixel and a smaller contribution from distant pixels.


## Image Derivatives {#image-derivatives}

We can use convolution to approximate the partial derivative (or finite difference) of an image. Recall that

\begin{equation\*}
\frac{\partial f}{\partial x} = \lim\limits\_{\epsilon \rightarrow 0} \frac{f(x + \epsilon, y) - f(x, y)}{\epsilon}.
\end{equation\*}

We can estimate this as a finite difference

\begin{equation\*}
\frac{\partial h}{\partial x} \approx h\_{i+1, j} - h\_{i-1, j}.
\end{equation\*}

Which can be applied via convolution given a kernal

\begin{equation\*}
\mathcal{H} =
\begin{bmatrix}
0 & 0 & 0\\\\
-1 & 0 & 1\\\\
0 & 0 & 0\\\\
\end{bmatrix}.
\end{equation\*}

Applying both the horizontal and vertical derivative kernels to an image to a simple square shows the detection of horizontal and vertical edges.

{{< figure src="/ox-hugo/2022-01-31_17-36-28_screenshot.png" caption="<span class=\"figure-number\">Figure 9: </span>Horizontal (right) and vertical (middle) derivative kernels applied to the original image (left)." >}}

The results of applying the derivative kernels are referred to as vertical edge and horizontal edge **scores**. Consider the middle image in the figure above. The left edge has pixel values of 255; the right edge has -255. In either case, a high absolute score reveals that there is an edge. These scores report the direction of greatest change. The 255 on the left edge indicates the direction is to the right, while the -255 score indicates the direction is to the left. All other instances of the image return a rate of change of 0.
Let's see how these filters perform on a more interesting image.

{{< figure src="/ox-hugo/2022-01-31_22-45-59_screenshot.png" caption="<span class=\"figure-number\">Figure 10: </span>Vertical derivative filter (left) and horizontal derivative filter (right)." >}}
