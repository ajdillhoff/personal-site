+++
title = "Image Segmentation"
authors = ["Alex Dillhoff"]
date = 2022-02-22T00:00:00-06:00
tags = ["gpgpu"]
draft = false
lastmod = 2024-02-15
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Resources](#resources)
- [Introduction](#introduction)
- [Defining Segmentations](#defining-segmentations)
- [Grouping](#grouping)
- [Segmentation Methods](#segmentation-methods)

</div>
<!--endtoc-->



## Resources {#resources}

-   <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/> (Berkeley Segmentation Database)
-   <https://arxiv.org/abs/2105.15203v2> (SegFormer)
-   <https://arxiv.org/abs/1703.06870> (Mask R-CNN)
-   <https://github.com/sithu31296/semantic-segmentation> (Collection of SOTA models)


## Introduction {#introduction}

Feature extraction methods such as [SIFT]({{< relref "scale_invariant_feature_transforms.md" >}}) provide us with many distinct, low-level features that are useful for providing local descriptions images.
We now "zoom out" and take a slightly higher level look at the next stage of image summarization.
Our goal here is to take these low-level features and group, or fit, them together such that they represent a higher level feature.
For example, from small patches representing color changes or edges, we may wish to build higher-level feature representing an eye, mouth, and nose.

{{< figure src="Introduction/2022-02-23_08-12-56_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Capsule networks learn template components that make up handwritten images (Kosiorek et al.)" >}}

The goal of **image segmentation** is to obtain a compact representation of distinct features in an image.
We will see that this task is loosely defined and will vary from application to application.
For example, in one task we may wish to segment individual people detected in an image.
In another task, we may wish to segment the clothes they are wearing to identify certain fashion trends or make
predictions about the time of year based on an image.

{{< figure src="Introduction/2022-02-22_09-22-07_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Mask R-CNN results on COCO dataset (He et al.)" >}}

Popular methods for segmenting images use clustering or graph techniques.
Clustering methods will group local pixels together into regions based on similarity.
There are also a class of segmentation based on locating boundary curves called [Active Contours]({{< relref "active_contours.md" >}}).
These segment distinct boundaries based on image features such as edges and are motivated by physical or mathematical properties.

{{< figure src="Introduction/2022-02-22_09-32-59_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Comparison results from SegFormer paper (Xie et al.)" >}}


## Defining Segmentations {#defining-segmentations}

The task of image segmentation comes with ambiguity.
People will segment images in different ways depending on the task.


## Grouping {#grouping}

At the pixel-level, features such as edges will look similar regardless of the context.
An edge segment from a car will look similar to that of a building at a small enough scale.
As soon as those features are grouped together, their representation completely changes.
A collection of edges becomes a square or a corner.
The higher-level of grouping we have, the more that the collection of features begin to diverge from each other.

[ConvNet Gradient Visualization](https://www.katnoria.com/gradvizv1/)

Consider the famouse Muller-Lyer illusion seen below.

{{< figure src="/ox-hugo/2022-02-22_20-41-36_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>From Forsyth and Ponce \"Computer Vision - A Modern Approach\"" >}}

Looking at the details individually, it is easier to see that the lines are the same length.
When considering the grouped features around them, they appear different.

Another view of segmentation considers what is the figure versus what is the ground, or background.

{{< figure src="/ox-hugo/2022-02-22_20-49-46_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>From Forsyth and Ponce \"Computer Vision - A Modern Approach\"" >}}

The [Gestalt school of psychology](https://en.wikipedia.org/wiki/Gestalt_psychology) posited that grouping was the key to visual understanding.
Towards establishing a theory of segmentation, a set of factors was proposed:

1.  **Proximity** - Group elements that are close together.
2.  **Similarity** - Elements that share some sort of measurable similarity are grouped.
3.  **Common fate** - Often tied to temporal features, elements with a similar trajectory are grouped.
4.  **Common region** - Elements enclosed in a region are part of the same group. This region could be arbitrary.
5.  **Parallelism** - Parallel elements are grouped.
6.  **Closure** - Lined or curves that are closed.
7.  **Symmetry** - Elements exhibiting some sort of symmetry. For example, a mirrored shaped.
8.  **Continuity** - Continuous curves are grouped.
9.  **Familiar configuration** - Lower level elements, when grouped together, form a higher level object.

{{< figure src="/ox-hugo/2022-02-22_21-03-09_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>From Forsyth and Ponce \"Computer Vision - A Modern Approach\"" >}}

{{< figure src="/ox-hugo/2022-02-22_21-04-44_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>From Forsyth and Ponce \"Computer Vision - A Modern Approach\"" >}}

{{< figure src="/ox-hugo/2022-02-24_18-23-16_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>Sky and Water by M.C. Escher" >}}

Intuiting applications for some of these rules is easier than others.
For example, familiar configuration suggests that some familiar object can be identified by the sum of its parts.
This is especially helpful for problems where the whole object is occluded.

{{< figure src="/ox-hugo/2022-02-22_21-09-15_screenshot.png" caption="<span class=\"figure-number\">Figure 9: </span>Parts of the hand are occluded either by the hand itself or some object (Mueller et al.)" >}}

Common fate is a useful rule when considering tracking an object or group of objects over a series of frames.
Even something a simple as frame differencing is an efficient preprocessing step to removing unrelated information.


## Segmentation Methods {#segmentation-methods}

-   [Active Contours]({{< relref "active_contours.md" >}})
-   [Segmentation via Clustering]({{< relref "segmentation_via_clustering (2).md" >}})
