+++
title = "Sampling and Aliasing"
authors = ["Alex Dillhoff"]
date = 2022-01-30T00:00:00-06:00
tags = ["computer vision"]
draft = false
sections = "Computer Vision"
lastmod = 2025-01-21
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Resizing](#resizing)
- [Sampling](#sampling)

</div>
<!--endtoc-->



## Resizing {#resizing}

-   Aliasing arises through resampling an image
-   How to resize - algorithm
-   How to resolve aliasing

Resizing an image, whether increase or decreasing the size, is a common image operation. In Linear Algebra, **scaling** is one of the transformations usually discussed, along with rotation and skew. Scaling is performed by creating a transformation matrix

\begin{equation\*}
M =
\begin{bmatrix}
s & 0\\\\
0 & s
\end{bmatrix},
\end{equation\*}

where \\(s\\) is the scaling factor. This matrix can then be used to transform the location of each point via matrix multiplication.

Is is that simple for digital images? Can we simply transform each pixel location of the image using \\(M\\)? There are a couple of steps missing when it comes to scaling digital images. First, \\(M\\) simply creates a mapping between the location in the original image and the corresponding output location in the scaled image. If we were to implement this in code, we would need to take the pixel's value from the original image.


### A Simple example {#a-simple-example}

Take a \\(2 \times 2\\) image whose pixel values are all the same color.

{{< figure src="/ox-hugo/2022-01-29_21-24-11_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>2 by 2 image whose values are the same." >}}

If we transform each pixel location of the image and copy that pixel's value to the mapped location in the larger image, we would get something as seen in the figure below.

{{< figure src="/ox-hugo/2022-01-29_21-29-02_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>The resulting scaled image." >}}

This image is exactly what we would expect. The resulting image is two times as large as the first. What pixel values should the new ones take on? This is a question of **sampling**.


## Sampling {#sampling}

Given \\(s = 2\\), the scaling matrix maps the original pixel locations to their new values:

-   \\((0, 0) \mapsto (0, 0)\\)
-   \\((0, 1) \mapsto (0, 2)\\)
-   \\((1,0) \mapsto (2, 0)\\)
-   \\((1,1) \mapsto (2, 2)\\)

What values should be given to the unmapped values of the new image? There are several sampling strategies used in practice. Two of the most common approaches are **nearest neighbor** and **bilinear** sampling. Let's start with the nearest neighbor approach.


### Nearest Neighbor {#nearest-neighbor}

First, we let the pixel location in an image be the center of that pixel, as depicted below.

{{< figure src="/ox-hugo/2022-01-29_22-50-00_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>A 2-by-2 image with pixel locations depicted as dots in the center." >}}

To establish a map between the pixel locations in the scaled image and that of the original image, we shrink the grid on the larger image and superimpose it over the smaller image.

{{< figure src="/ox-hugo/2022-01-29_22-51-11_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Pixel grid of larger image superimposed on original image." >}}

With nearest neighbor interpolation, the pixel value in the resized image corresponds to that of the nearest pixel in the original image. In the above figure, we can see that pixels \\((0, 0), (0, 1), (1, 0), \text{ and } (1, 1)\\) in the resized image are closest to pixel \\((0, 0)\\) in the original image. Thus, they will take on that pixel's value.

Let's compare both of these approaches on a real image. The first figure below shows the original image \\((16 \times 16\\)). The following figure shows the image resized to \\((32 \times 32)\\) with no interpolation and nearest neighbor interpolation, respectively.

{{< figure src="/ox-hugo/2022-01-29_23-59-17_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>Original image." >}}

{{< figure src="/ox-hugo/2022-01-29_23-59-52_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>Image resized with no interpolation (left) and nearest neighbor (right)." >}}


### Bilinear Interpolation {#bilinear-interpolation}

**Bilinear interpolation** is a slightly more sophisticated way of sampling which takes into account all neighboring pixels in the original image. The value of the pixel in the sampled image is a linear combination of the values of the neighbors of the corresponding pixel it is mapped to.

Consider a \\(3 \times 3\\) image upsampled to a \\(8 \times 8\\) image. The figure below shows the original image with the coordinates of the upsampled image superimposed on it.

{{< figure src="/ox-hugo/2022-01-30_11-36-20_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>3-by-3 grid with 8-by-8 coordinates overlaid." >}}

**How do we determine the coordinate map between the original and upscaled image?**
Solve a linear system.

Note the extreme values of the image. That is, the smallest and largest coordinates. Since we stated previously that \\((0, 0)\\) refers to the coordinate in the middle of the pixel, the top-left of the image boundary for any image is \\((-0.5, -0.5)\\). The bottom-right corner for the smaller image is \\((2.5, 2.5)\\). The bottom-right corner for the resized image is \\((7.5, 7.5)\\). The equation that maps the top-left coordinates between the images is given by

\\[
-\frac{1}{2} = -\frac{1}{2}a + b.
\\]

The equation that maps the bottom-right coordinates between the images is given by

\\[
\frac{5}{2} = \frac{15}{2}a + b.
\\]

Thus, we have 2 equations 2 unknowns. Solving this yields \\(a = \frac{3}{8}\\) and \\(b = -\frac{5}{16}\\).

With the mapping solved, let's compute the color value for pixel \\((3, 3)\\) in the upsampled image. Here, \\((3, 3) \mapsto (\frac{13}{16}, \frac{13}{16})\\) in the original image. Our problem for this particular pixel is reduced to the following figure.

{{< figure src="/ox-hugo/2022-01-30_18-08-02_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>Determining the pixel value for the mapped pixel using bilinear interpolation." >}}

We first interpolate between two pairs of pixels in the original image. That is, we find \\(p\_1\\) and \\(p\_2\\) in the following figure.

{{< figure src="/ox-hugo/2022-01-30_18-15-27_screenshot.png" caption="<span class=\"figure-number\">Figure 9: </span>Step 1: Interpolate the pixel values between two pixels for \\(p\_1\\) and \\(p\_2\\)." >}}

Here, \\(p\_2 = \frac{3}{16}(255, 255, 255) + \frac{13}{16}(128, 128, 128) \approx (152, 152, 152)\\) and \\(p\_1 = \frac{3}{16}(255, 0, 0) + \frac{13}{16}(255, 255, 255) \approx (255, 207, 207)\\). Note that the contribution of the pixel depends on the weight on the other side the intermediate value \\(p\_i\\). For example, if you think of \\(p\_1\\) as a slider from the red pixel to the white pixel. The value to the left of the slider reflects the contribution of the pixel to the right, and vice versa.

{{< figure src="/ox-hugo/2022-01-30_18-23-43_screenshot.png" caption="<span class=\"figure-number\">Figure 10: </span>Computed values of \\(p\_1\\) and \\(p\_2\\)." >}}

Finally, the value of the new pixel is a linear combination of \\(p\_1\\) and \\(p\_2\\). That is \\(p = \frac{13}{16}(152, 152, 152) + \frac{3}{16}(255, 207, 207) \approx (171, 162, 162)\\).

{{< figure src="/ox-hugo/2022-01-30_18-28-03_screenshot.png" caption="<span class=\"figure-number\">Figure 11: </span>The final pixel value computed from \\(p\_1\\) and \\(p\_2\\)." >}}

The following figure compares the original image with both types of interpolation discussed in this section.

{{< figure src="/ox-hugo/2022-01-30_18-34-13_screenshot.png" caption="<span class=\"figure-number\">Figure 12: </span>Original image (left), upscaled 2x with NN interpolation (middle), upscaled with bilinear interpolation (right)." >}}


### Aliasing {#aliasing}

Nearest neighbor interpolation often leads to images with **aliasing**. In general, aliasing occurs when two signals are sampled at such a frequency that they become indistinguishable from each other. Usually, images are smoothed prior to upsampling or downsampling in an effort to alleviate the effects of aliasing. The figure below shows a downsampled image using nearest neighbor interpolation.

{{< figure src="/ox-hugo/2022-01-30_10-31-40_screenshot.png" caption="<span class=\"figure-number\">Figure 13: </span>Image downsampled by 4x. Notice the \"jaggies\", especially along straight lines." >}}

By blurring the image and using bilinear interpolation, the same image looks much smoother when downsized.

{{< figure src="/ox-hugo/2023-01-25_19-30-16_screenshot.png" caption="<span class=\"figure-number\">Figure 14: </span>Image downsampled by 4x using bilinear interpolation." >}}
