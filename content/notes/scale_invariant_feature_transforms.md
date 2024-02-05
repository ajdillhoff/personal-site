+++
title = "Scale Invariant Feature Transforms"
authors = ["Alex Dillhoff"]
date = 2022-01-22T00:00:00-06:00
tags = ["computer vision"]
draft = false
lastmod = 2024-01-28
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Difference of Gaussians](#difference-of-gaussians)
- [Keypoint Localization](#keypoint-localization)
- [Orientation Assignment](#orientation-assignment)
- [Descriptor Formation](#descriptor-formation)

</div>
<!--endtoc-->



## Introduction {#introduction}

<https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf>

General approach to computing SIFT features:

1.  Scale-space Extrema Detection
2.  Keypoint localization
3.  Orientation Assignment
4.  Generate keypoint descriptors


## Difference of Gaussians {#difference-of-gaussians}

This same technique for detecting interesting points in a scale-invariant way can be approximated by taking the **Difference of Gaussians**. Consider the figure below.

{{< figure src="/ox-hugo/2022-02-10_10-31-26_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Comparison of DoG and Laplacian. Credit: Fei-Fei Li." >}}

By taking the difference of images smoothed by a Gaussian with different values of \\(\sigma\\), the resulting pixel values correspond to areas with high gradient norms in the less blurry version.

Let \\(I\_{\sigma\_1}\\) be the image blurred with a smaller value of \\(\sigma\\) and \\(I\_{\sigma\_2}\\) be the image blurred with a larger value.
Then \\(D(I\_{\sigma\_1}, I\_{\sigma\_2}) = I\_{\sigma\_2} - I\_{\sigma\_1}\\).
If a region in \\(I\_{\sigma\_1}\\) is locally flat, it will also be in flat in \\(I\_{\sigma\_2}\\).
The difference will be relatively small for that region.
If there are abrupt changes in a local region within \\(I\_{\sigma\_1}\\), they will be smoothed in \\(I\_{\sigma\_2}\\).
Therefore, the difference \\(D(I\_{\sigma\_1}, I\_{\sigma\_2})\\) will be higher for that region.

{{< figure src="/ox-hugo/2022-02-13_20-29-08_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>The Royal Concertgebouw in Amsterdam." >}}

{{< figure src="/ox-hugo/2022-02-13_20-41-50_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Difference of Gaussian between the original image blurred with \\(\sigma = 0.5\\) and \\(\sigma=1.5\\)." >}}

When building SIFT features, the extremum are selected by comparing 3 DoG images.
These are selected by evaluating each pixel to 26 of its neighbors in the current scale space and neighboring DoG spaces as visualized below.

{{< figure src="/ox-hugo/2022-02-13_18-50-20_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Finding extrema of pixel (i, j) in a neighborhood of 26 values (<a href=\"#citeproc_bib_item_2\">Lowe 2004</a>)." >}}

To build the DoG pyramid, the authors propose that images are separated by a constant factor \\(k\\) in scale space.
Each octave of scale space is divided such that the scalespace doubles every \\(s\\) samples.

Starting with \\(\sigma = 0.5\\), if we choose \\(s=3\\) then the fourth sample will be at \\(\sigma = 1\\), the seventh at \\(\sigma=2\\), and so on.
To make sure the DoG images cover the full range of an octave, \\(s + 3\\) images need to be created per octave.

**Why \\(s + 3\\)?**

Each octave should evaluate local extrema for \\(s\\) scales.
To evaluate this for scale \\(\sigma\_s\\), we need the DoG for scales \\(\sigma\_{s-1}\\) and \\(\sigma\_{s+1}\\).
This would require 4 Gaussians images to compute.
The figure below represents the stack for \\(s=2\\).

{{< figure src="/ox-hugo/2022-02-10_17-35-51_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>DOG figure (<a href=\"#citeproc_bib_item_2\">Lowe 2004</a>)." >}}

**How is the value of \\(s\\) determined?**

In the paper, the authors perform a repeatability test to determine if the keypoints would be localized even with random augmentations. The process is as follows:

1.  Randomly augment an input image with noise, color jitter, scale, rotation, etc.
2.  Compute keypoints using using the extrema detection.
3.  Compare detected keypoints with known keypoints from original samples.

The authors found that using \\(s = 3\\) provided the highest percentage of repeatability in their experiments.

{{< figure src="/ox-hugo/2022-02-13_19-25-02_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>Measuring repeatability of keypoint detections versus # of scales sampled per octave (<a href=\"#citeproc_bib_item_2\">Lowe 2004</a>)." >}}


## Keypoint Localization {#keypoint-localization}

Given the candidate keypoints selected by picking out local extrema, they pool of responses can further be refined
by removing points that are sensitive to noise or located along an edge. They borrow the same approach used in the Harris corner detector to select more robust interest points in corners.

{{< figure src="/ox-hugo/2022-02-15_20-36-13_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>Refinement of candidate keypoints by filtering those sensitive to noise (c) and those representing ambiguity along edges (d) (<a href=\"#citeproc_bib_item_2\">Lowe 2004</a>)." >}}


## Orientation Assignment {#orientation-assignment}

Given a keypoint, an orientation histogram is generated. The authors use 36 bins to cover a 360 degree range for orientations.
Similar to [Histogram of Oriented Gradients]({{< relref "histogram_of_oriented_gradients.md" >}}), the orientations are weighted by their gradient magnitudes (<a href="#citeproc_bib_item_1">Dalal and Triggs 2005</a>).
Additionally, a Gaussian-weighted circular patch is applied, centered on the keypoint, to further weight the responses.
This means that points farther away from the center contribute less to the overall feature vector.

In order to make the keypoint rotation invariant, the dominant orientation is determined.
If there are orientations that are within 80% of the highest orientation peak, multiple keypoints will be created using those orientations as well.

Orientations in this window are rotated by the dominant gradient so that all directions are with respect to the dominant orientation.
This is a more efficient alternative to rotating the entire image by that orientation.


## Descriptor Formation {#descriptor-formation}

{{< figure src="/ox-hugo/2022-02-15_20-22-39_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>Keypoint descriptor generation (<a href=\"#citeproc_bib_item_2\">Lowe 2004</a>)." >}}

In the paper, the authors generate keypoints using a \\(16 \times 16\\) window from which \\(4 \times 4\\) descriptors are generated following the descriptions above.
Through experimentation, each \\(4 \times 4\\) descriptor uses 8 orientations, resulting in a feature vector \\(\mathbf{x} \in \mathbb{R}^{128}\\).

Different levels of contrast will product edges with higher gradient magnitudes.
To account for this, the final feature vector is normalized using the \\(L2\\) hysteresis approach used in Harris corner detection.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Dalal, N., and B. Triggs. 2005. “Histograms of Oriented Gradients for Human Detection.” In <i>2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05)</i>, 1:886–93 vol. 1. <a href="https://doi.org/10.1109/CVPR.2005.177">https://doi.org/10.1109/CVPR.2005.177</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Lowe, David G. 2004. “Distinctive Image Features from Scale-Invariant Keypoints.” <i>International Journal of Computer Vision</i> 60 (2): 91–110. <a href="https://doi.org/10.1023/B:VISI.0000029664.99615.94">https://doi.org/10.1023/B:VISI.0000029664.99615.94</a>.</div>
</div>
