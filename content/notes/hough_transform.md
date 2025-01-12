+++
title = "Hough Transform"
authors = ["Alex Dillhoff"]
date = 2022-02-17T00:00:00-06:00
tags = ["computer vision"]
draft = false
lastmod = 2024-02-11
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Algorithm](#algorithm)
- [Rectangle Detection based on a Windowed Hough Transform](#rectangle-detection-based-on-a-windowed-hough-transform)

</div>
<!--endtoc-->



## Introduction {#introduction}

Fitting a model to a set of data by consensus, as in [RANdom SAmple Consensus]({{< relref "random_sample_consensus.md" >}}), produces a parameter estimate that is robust to outliers. A similar technique for detecting shapes in images is the **Hough Transform**.
Originally it was designed for detecting simple lines, but it can be extended to detect [arbitrary shapes](https://en.wikipedia.org/wiki/Generalised_Hough_transform).

The transform is computed given an edge image. Each edge pixel in the image casts a vote for a line given a set of parameters.
This vote is added to an accumulator array which tallies the votes over all pixels for all parameter choices.


## Algorithm {#algorithm}

An accumulator array holds the votes for each edge point.
The indices to this array represent the line parameters \\((\rho, \theta)\\), where

\\[
\rho = x \cos \theta + y \sin \theta.
\\]

{{< figure src="/ox-hugo/2022-02-17_16-47-33_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Parameterization of a line in a Hough transform. Source: Wikipedia" >}}

In this parameterization, \\((\rho, \theta)\\) represents a vector that is normal to the line.
For each edge pixel in the original image, a range of \\(\theta\\) values are tested.
The resolution of \\(\theta\\) values used is set as a hyperparameter to the algorithm.
A vote is cast for every single line within the resolution of \\(\theta\\) by incrementing the accumulator array at entry \\((\rho, \theta)\\).

{{< figure src="/ox-hugo/2022-02-17_16-54-33_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Visualization of voting procedure. Source: Wikipedia" >}}

After all edge pixels are evaluated, the accumulator array must be post processed to select the lines with the most agreement.
Since the accumulator array is 2D, it can be visualized as seen below.

{{< figure src="/ox-hugo/2022-02-17_16-58-06_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Visualization of accumulator array. Source: Wikipedia" >}}


## Rectangle Detection based on a Windowed Hough Transform {#rectangle-detection-based-on-a-windowed-hough-transform}

As mentioned in the introduction, Hough transforms can be extended to detect arbitrary shapes.
In their [2004 publication](http://sibgrapi.sid.inpe.br/col/sid.inpe.br/banon/2004/08.03.17.14/doc/1.pdf), Jung and Schramm approach the problem of detecting rectangles in an image via Hough transforms.
By analyzing the spatial relationship of peaks in a standard Hough transform, rectangles can reliably be detected.


### Detecting Rectangles via Hough peaks {#detecting-rectangles-via-hough-peaks}

If we consider pixel located at the center of a rectangle, a few key symmetries can be observed.

1.  The detected peaks from the Hough window will be in pairs, equidistant from the center.
2.  Two pairs will be separated by \\(90^{\circ}\\) with respect to the \\(\theta\\) axis.
3.  The distance between the peaks of a pair are the sides of the rectangle.

{{< figure src="/ox-hugo/2022-02-19_16-33-06_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Rectangle centered at the origin." >}}

{{< figure src="/ox-hugo/2022-02-19_16-33-56_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>The resulting Hough transform of the previous figure." >}}


### The Algorithm {#the-algorithm}

Each pixel in the image is evaluated using a sliding window approach.
Consider a pixel centered on \\((x\_c, y\_c)\\).
The size of the window determines the maximum size of the rectangle that can be detected in a given window.
The authors use a circular threshold with a minimum and maximum diameter \\(D\_{min}\\) and \\(D\_{max}\\).
That is, the search region is a circle such that the smallest detectable rectangle has a side length of no less than \\(D\_{min}\\) and a diagonal length of no more than \\(D\_{max}\\). The figure below visualizes this region.

{{< figure src="/ox-hugo/2022-02-19_18-56-46_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>Search region based on a circle. Source: Jung and Schramm 2004." >}}

With the search region defined, a hough transform for that region is computed.
The paper mentions an optimization step when selecting the discretization steps \\(d\_{\theta}\\) and \\(d\_{\rho}\\).
If the input image is large, the resulting Hough transform will also be large.
They recommend picking \\(d\_{\theta} = \frac{3 \pi}{4 D\_{max}}\\) and \\(d\_{\rho} = \frac{3}{4}\\).

The Hough image is further processed in order to extra local extrema.
These peaks should correspond to the lines of the rectangle.
First, an enhanced image is created following

\\[
C\_{\text{enh}}(\rho, \theta) = h w \frac{C(\rho, \theta)^2}{\int\_{-h/2}^{h/2}\int\_{-w/2}^{h/2} C(\rho + y, \theta + x)dx dy}.
\\]

****How is such an integral computed?****
The integral is the "area under the curve" of a given signal. In this case, the accumulator image is a 2D signal for which a discrete approximation of the integral can be computed.
This can be implemented via convolution.
The local maxima of this enhanced image are those such that \\(C(\rho, \theta) \geq T\_C\\), where \\(T\_C = 0.5D\_{min}\\).

{{< figure src="/ox-hugo/2022-02-19_19-46-28_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>Original Hough image (left) and enhanced (right) (Jung and Schramm 2004)." >}}

With the detected peaks from the enhanced image, four peaks are selected with satisfy the symmetries of rectangles as listed above. This is shown as equation 3 in the paper:

\begin{align\*}
&\Delta \theta = |\theta\_i - \theta\_j| < T\_{\theta},\\\\
&\Delta \rho = |\rho\_i + \rho\_j| < T\_{\rho},\\\\
&|C(\rho\_{i}, \theta\_i) - C(\rho\_j, \theta\_j)| < T\_{L}\frac{C(\rho\_i, \theta\_i) + C(\rho\_j, \theta\_j)}{2}.
\end{align\*}

\\(T\_{\theta}\\) is used to threshold peaks corresponding to parallel lines (\\(\theta\_i \approx \theta\_j\\)).
\\(T\_{\rho}\\) is a threshold for symmetry (equal distance between lines and center).
\\(T\_L\\) ensures that line segments have approximately the same length.

For peaks that satisfy the equations above, an extended peak \\(P\_k = (\pm \xi, \alpha\_k)\\) is formed, where

\\[
\alpha\_k = \frac{1}{2}(\theta\_i + \theta\_j) \text{ and } \xi\_k = \frac{1}{2}|\rho\_i - \rho\_j|.
\\]

Finally, a rectangle is detected if the pairs of lines are orthogonal. That is, if

\\[
\Delta \alpha = ||\alpha\_k - \alpha\_l| - 90^{\circ}| < T\_{\alpha}.
\\]

The rectangle parameters are encoded by \\(\alpha\_k\\) being its orientation and \\(2\xi\_k\\) and \\(2\xi\_l\\) being its sides.

The final step in the paper is to remove duplicates since, depending on the threshold choices, multiple candidates for a rectangle may be detected.
The intuition behind this step is to create an error measure that summarizes how well the symmetries defined by the conditions required for a rectangle are respected.

\\[
E(P\_k, P\_l) = \sqrt{a(\Delta \theta\_k^2 + \Delta \theta\_l^2 + \Delta \alpha^2) + b(\Delta \rho\_k^2 + \Delta \rho\_l^2)}
\\]

In this error measure, \\(a\\) and \\(b\\) are used to weight the angular and distance errors differently since a change in 1 pixel would be more significant than a change of 1 degree.

If the difference in orientation for each line detected in the sides is greater, the orientation error increases.
Likewise, the more that the pairs of lines stray from orthogonality, the greater the error becomes.
For the distance measure, the more offset the lines are with respect to the center, the greater the error becomes.
Thus, the rectangle that meets these criteria best will have the lowest error.
