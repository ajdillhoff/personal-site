+++
title = "Image Segmentation"
authors = ["Alex Dillhoff"]
date = 2022-02-22T00:00:00-06:00
tags = ["computer vision"]
draft = false
lastmod = 2024-06-19
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Resources](#resources)
- [Introduction](#introduction)
- [Gestalt Theory](#gestalt-theory)
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

Feature extraction methods such as [SIFT]({{< relref "scale_invariant_feature_transforms.md" >}}) provide us with many distinct, low-level features that are useful for providing local descriptions images. We now "zoom out" and take a slightly higher level look at the next stage of image summarization.
Our goal here is to take these low-level features and group, or fit, them together such that they represent a higher level feature. For example, from small patches representing color changes or edges, we may wish to build higher-level feature representing an eye, mouth, and nose.

{{< figure src="/ox-hugo/2024-06-19_16-29-57_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Capsule networks learn template components that make up handwritten images (Kosiorek et al.)" >}}

The goal of **image segmentation** is to obtain a compact representation of distinct features in an image. We will see that this task is loosely defined and will vary from application to application. For example, in one task we may wish to segment individual people detected in an image. In another task, we may wish to segment the clothes they are wearing to identify certain fashion trends or make predictions about the time of year based on an image.

{{< figure src="/ox-hugo/2024-06-19_16-30-13_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Mask R-CNN results on COCO dataset (He et al.)" >}}

Image segmentation is one of the oldest tasks in computer vision and, as such, encompasses many methods and techniques. The following list of segmentation categories is not exhaustive, but covers many notable approaches.

1.  **Region:** Methods that group by regions and either grow, merge, or spread them fall into this category. A simple region-based approach would be to apply morphological operations on thresholded images.
2.  **Boundary:** Separating parts of the image based on edges, lines, or specific points.
3.  **Clustering:** Methods that utilize some form of clustering such as K-means or Gaussian models to group pixels fall into this category.
4.  **Watershed:** Groups pixels by treating the image as a topographical map. Consider a large pool with 4 local minima such that there are 4 tiny pools inside it. As you fill the entire pool up, eventually the water line from each of the 4 smaller pools will touch, creating a boundary between different regions.
5.  **Graph-Based:** Segmentation algorithms based on graph theory.
6.  **Mean-Shift:** Clustering methods fall into this category.
7.  **Normalized Cuts:** Quantifies the _strength_ of connectedness between groups before separating weak connections.

Each of these styles can be used for general segmentation or a well-defined downstream task. General segmentation may be used for unsupservised segmentation of an image for the purpose of creating superpixels (<a href="#citeproc_bib_item_1">Achanta et al. 2012</a>). A common downstream task is **semantic segmentation**, where distinct objects in an image are segmented. An even more difficult version of semantic segmentation is **instance segmentation** where multiple instances of objects are labelled separately.

{{< figure src="/ox-hugo/2024-06-19_16-30-36_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Comparison results from SegFormer paper (Xie et al.)" >}}


## Gestalt Theory {#gestalt-theory}

Gestalt psychology provides a theory of perception emphasizing grouping patterns. Examining different ways to think about grouping sheds light on the open-endedness of segmentation in general.


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
-   [Segmentation via Clustering]({{< relref "segmentation_via_clustering.md" >}})

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Achanta, Radhakrishna, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk. 2012. “SLIC Superpixels Compared to State-of-the-Art Superpixel Methods.” <i>Ieee Transactions on Pattern Analysis and Machine Intelligence</i> 34 (11): 2274–82. <a href="https://doi.org/10.1109/TPAMI.2012.120">https://doi.org/10.1109/TPAMI.2012.120</a>.</div>
</div>
