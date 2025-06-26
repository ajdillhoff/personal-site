+++
title = "Segmentation via Clustering"
authors = ["Alex Dillhoff"]
date = 2022-02-24T00:00:00-06:00
tags = ["computer vision"]
draft = false
lastmod = 2025-06-24
sections = "Computer Vision"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Agglomerative Clustering](#agglomerative-clustering)
- [K-Means Clustering](#k-means-clustering)
- [Mean-Shift Clustering](#mean-shift-clustering)
- [Simple Linear Iterative Clustering (SLIC)](#simple-linear-iterative-clustering--slic)
- [Superpixels in Recent Work](#superpixels-in-recent-work)

</div>
<!--endtoc-->



## Introduction {#introduction}

The goal of segmentation is fairly broad: group visual elements together.
For any given task, the question is _how are elements grouped?_
At the smallest level of an image, pixels can be grouped by color, intensity, or spatial proximity.
Without a model of higher level objects, the pixel-based approach will break down at a large enough scale.

Segmentation by thresholding works in cases where the boundaries between features are clearly defined.
However, thresholding is not very robust to complex images with noise.
Consider a simple image and its intensity histogram as noise is added.

{{< figure src="/ox-hugo/2022-03-01_10-27-51_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>From left to right, a noiseless image with increasing amounts of Gaussian noise added. Source: Pearson Education, Inc." >}}

Even with some noise added, as seen in the middle image, thresholding is still relatively straightforward.
Once enough noise is added, thresholding via pixel intensities will not work.
A more sophisticated approach is needed in this case.

Clustering is a fairly intuitive way to think about segmentation.
Instead of a fine-grained representation of an image as a collection of pixels, it is represented as groups or clusters that share some common features.
The general process of clustering is simple.
The image is represented as a collection of feature vectors (intensity, pixel color, etc.).
Feature vectors are assigned to a single cluster. These clusters represent some segment of the image.

When it comes to clustering methods, there are two main approaches: agglomerative and divisive.
Simply, one is a bottom-up approach. The other is a top-down approach.
After briefly introductin agglomerative clustering, we will explore specific implementations of segmentation using k-means clustering as well as segmentation using superpixels (<a href="#citeproc_bib_item_1">Achanta et al. 2012</a>).


## Agglomerative Clustering {#agglomerative-clustering}

Agglomerative clustering methods start by assuming every element is a separate cluster.
Elements are formed based on some local similarities.
As these methods iterate, the number of clusters decreases.
Deciding which elements to merge depends on **inter-cluster distance**.
The exact choice of distance is dependent on the task. Some examples include:

1.  **Single-link clustering**: The distance between the closest elements.
2.  **Complete-link clustering**: The maximum distance between an element of the first cluster and one of the second.
3.  **Group average clustering**: Average distance of elements in a cluster.

**How many clusters are or should be in a single image?**

This is a difficult question to answer for many reasons. The answer will be largely dependent on the task at hand.
It is a problem of learning the underlying generative process of the visual elements in the image.
By defining the specific goal of segmentation (segment by color, shape, etc.), we are introducing a prior about the underlying generative processes which formed the image.

<a id="figure--fig1"></a>

{{< figure src="/ox-hugo/2022-02-24_16-24-28_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>3D-PointCapsNet learns point segmentations on only 1% of the training data (Zhao et al.)." >}}

There are approaches which attempt to segment objects in semi-supervised settings.
As seen in [Figure 1](#figure--fig1), Zhao et al. propose a part segmentation model for 3D objects which only utilizes 1-5% of the training part labels (<a href="#citeproc_bib_item_5">Zhao et al. 2019</a>).

For example, if we divised an algorithm that would segment an image by color values, it might be able to segment the hand wearing a solid color glove relatively easily.
If we wanted to segment the hand into its individual joints, we would have to introduce a visual prior such as asking the subject to wear a multicolored glove.
We could also add prior information about the hand shape and joint configuration into the model itself.

<a id="figure--joint-pc"></a>

{{< figure src="/ox-hugo/2022-03-01_22-08-18_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>An image-based joint regression model predicts joint locations (left) along with a point cloud generated from the joint estimates (right)." >}}

In the [figure above](#figure--joint-pc), the kinematic hand model could be used to segment the hand by assigning points in the point cloud to the nearest joint as estimated by the model.

One way to visualize the cluster relationships is a _dendrogram_.
Initially, each element is its own cluster. As the process evolves and clusters are merged based on some similarity,
the hierarchy is updated to show how the connections are formed.

{{< figure src="/ox-hugo/2022-02-24_18-16-31_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Example output from scikit-image." >}}


## K-Means Clustering {#k-means-clustering}

-   [K-Means Variant KNN Demo](https://huggingface.co/spaces/SLU-CSCI5750-SP2022/homework03_DigitClassificationKNN)
-   <https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py>

K-Means clustering is a popular machine learning method used in both supervised and unsupervised settings.
It works by iteratively updating a set of _centroids_ or means until some stopping criteria is achieved.

To use this with image segmentation, we start by treating our image features as vectors.
In the RGB case, each pixel is a vector of 3 values.
It starts out by initializing \\(k\\) clusters randomly with means \\(\mathbf{m}\_i\\).
The next step is to compute the distance between the clusters and each point in the image.
Points are assigned to the cluster that is closest.

\\[
\text{arg}\min\_{C} \sum\_{i=1}^k \sum\_{\mathbf{z}\in C\_i}\\|\mathbf{z} - \mathbf{m}\_i\\|^2,
\\]

where \\(C = \\{C\_1, \dots, C\_k\\}\\) is the cluster set.

K-Means uses Expectation Maximization to update its parameters.
That is, it first computes the expected values given its current cluster centers before updating the cluster centers based on the new assignments.
The standard algorithm is as follows:

1.  **Initialize clusters** - Randomly select \\(k\\) points as cluster centers \\(\mathbf{m}\_i\\).
2.  **Assign samples to clusters** - Assign each sample to the closest cluster center based on some distance metric.
3.  **Update the means** - Compute a new value for the cluster centers based on the assignments in the previous step.
    \\[
       \mathbf{m}\_i = \frac{1}{|C\_i|}\sum\_{\mathbf{z} \in C\_i}\mathbf{z}, \quad i = 1, \dots, k
       \\]
4.  **Test for convergence** - Compute the distances between the means at time \\(t\\) and time \\(t - 1\\) as \\(E\\). Stop if the difference is less than some threshold: \\(E \leq T\\).

{{< figure src="/ox-hugo/2022-03-01_21-41-58_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>Image segmented using k-means with k=3. Source: Pearson Education, Inc." >}}


## Mean-Shift Clustering {#mean-shift-clustering}

K-Means requires us to define the number of clusters. That could be fine in applications where we need to balance our approach based on the requirements, but it leaves the method feeling less automated than we would hope for unsupervised learning.

Mean-shift clustering obviates the choice of clusters by treating every individual data point as a cluster center. As the algorithm progresses, the cluster copies are shifted towards regions of high density in the feature space. Points that end up at the same point are assigned to the same cluster.

{{< figure src="/ox-hugo/2025-06-24_10-19-02_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>Mean-Shift in action ([Source](https://ailephant.com/how-to-program-mean-shift/))" >}}


### Algorithm {#algorithm}

1.  **Initialize clusters** - Start with each point as a cluster center.
2.  **Shift clusters** - For each cluster center, compute the mean of all points within a certain radius \\(r\\).
    \\[\mathbf{m}\_i = \frac{1}{|C\_i|}\sum\_{\mathbf{z} \in C\_i}\mathbf{z}, \quad i = 1, \dots, k\\]
    where \\(C\_i\\) is the set of points within radius \\(r\\) of cluster center \\(\mathbf{m}\_i\\).
3.  **Translate window** - Move the cluster center to the computed mean.
4.  **Test for convergence** - If the cluster centers do not change significantly, stop. Otherwise, return to step 2.


### Application: Image Segmentation {#application-image-segmentation}

Mean-shift clustering can be used for image segmentation by treating each pixel as a point in the feature space. The kernel used for the mean-shift algorithm can be based on color and spatial proximity, allowing for the segmentation of regions in the image that share similar colors and are spatially close.

\\[
K\_{h\_s,h\_r}(\mathbf{x}) = \frac{C}{h\_s^2 h\_r^2}k\left(\left\\|\frac{\mathbf{x}^s}{h\_s}\right\\|^2\right)k\left(\left\\|\frac{\mathbf{x}^r}{h\_r}\right\\|^2\right)
\\]

where \\(h\_s\\) is the spatial bandwidth, \\(h\_r\\) is the color bandwidth, and \\(k\\) is a kernel function (e.g., Gaussian). A qualitative comparison of varying \\(h\_s\\) and \\(h\_r\\) is shown below.

{{< figure src="/ox-hugo/2025-06-26_10-10-41_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>Result of varying the spatial and color bandwidth (<a href=\"#citeproc_bib_item_2\">Comaniciu and Meer 2002</a>)." >}}

{{< figure src="/ox-hugo/2025-06-26_10-03-23_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>First two dimensions of LUV space (left) and the result of mean-shift (right) (<a href=\"#citeproc_bib_item_2\">Comaniciu and Meer 2002</a>)." >}}

Once an image has been processed via Mean-shift clustering, the region boundaries can be drawn arond each segment.

{{< figure src="/ox-hugo/2025-06-26_10-13-51_screenshot.png" caption="<span class=\"figure-number\">Figure 9: </span>After running mean-shift, the regions can be segmented (<a href=\"#citeproc_bib_item_2\">Comaniciu and Meer 2002</a>)." >}}


## Simple Linear Iterative Clustering (SLIC) {#simple-linear-iterative-clustering--slic}

Simple Linear Iterative Clustering (SLIC) is widely used algorithm based on K-Means clustering for image segmentation (<a href="#citeproc_bib_item_1">Achanta et al. 2012</a>).

As discussed in the original paper, the authors state that SLIC has two main advantages over traditional K-Means:

1.  The search space for assigning points is reduced, leading to an increase in performance.
2.  By weighting the distance measure, color and spatial proximity are both considered when forming clusters.

The algorithm itself is simple to understand and implement, as seen below.

{{< figure src="/ox-hugo/2022-03-01_23-39-33_screenshot.png" caption="<span class=\"figure-number\">Figure 10: </span>SLIC Algorithm (Achanta et al.)" >}}


### Initialization {#initialization}

To keep the search space smaller, the individual search regions are spaced \\(S = \sqrt{N/k}\\) pixels apart, where \\(N\\) is the number of pixels and \\(k\\) is the number of cluster centers.

The image itself is represented in [CIELAB color space](https://en.wikipedia.org/wiki/CIELAB_color_space).
This color space was chosen because it is _perceputally uniform_.
That is, it is useful for detecting small differences in color.

Each of the \\(k\\) pixel clusters is then defined as a superpixel consisting of the CIELAB color and position:

\\[
C\_i = [l\_i\ a\_i\ b\_i\ x\_i\ y\_i]^T.
\\]

For stability, the seed locations are moved to the lowest gradient position in a \\(3 \times 3\\) neighborhood.
If the superpixels are building locally distinct regions, it is better to avoid placing them on an edge (boundary) pixel.


### Search Space and Distance {#search-space-and-distance}

The search space for a cluster center is a region \\(2S \times 2S\\) around the cluster.
Each pixel in this region is compared to the cluster center \\(C\_k\\) using a distance measure \\(D\\).

The distance measure should consider both the spatial and color distances:

\begin{align\*}
d\_c &= \sqrt{(l\_j - l\_i)^2 + (a\_j - a\_i)^2 + (b\_j - b\_i)^2}\\\\
d\_s &= \sqrt{(x\_j - x\_i)^2 + (y\_j - y\_i)^2}\\\\
D' &= \sqrt{\Big(\frac{d\_c}{N\_c}\Big)^2 + \Big(\frac{d\_s}{N\_s}\Big)^2}
\end{align\*}

The individual distances should be normalized by their respective maximums since the range of CIELAB values is different from the variable maximum of \\(N\_s\\), which is based on the image size.
Here, \\(N\_s\\) corresponds to the sampling size \\(\sqrt{N/k}\\).

The authors found that normalizing this way was inconsistent since the color distances vary greatly from cluster to cluster.
They turn this normalization into a hyperparameter constant \\(m\\) so that the user can control the importance between spatial and color proximity.

\\[
D = \sqrt{d\_c^2 + \Big(\frac{d\_s}{S}\Big)^2 m^2}
\\]

A smaller \\(m\\) results in superpixels that adhere more to image boundaries, where a larger value promotes compact superpixels.


### Results {#results}

{{< figure src="/ox-hugo/2022-03-03_20-24-07_screenshot.png" caption="<span class=\"figure-number\">Figure 11: </span>Comparison of SLIC against other superpixel methods (Achanta et al.)" >}}

{{< figure src="/ox-hugo/2022-03-03_20-26-00_screenshot.png" caption="<span class=\"figure-number\">Figure 12: </span>Images segmented using a varying number of clusters (Achanta et al.)" >}}


## Superpixels in Recent Work {#superpixels-in-recent-work}

Superpixels are useful for reducing the dimensionality of the feature space.
Their applications include tracking, segmentation, and object detection.
Methods that extract superpixels do not work out of the box with deep learning methods
due to their non-differentiable formulation.
Deep learning methods rely on gradient descent to optimize their parameters.
This requires that the functions used in a deep network be differentiable.

{{< figure src="/ox-hugo/2022-03-03_20-47-51_screenshot.png" caption="<span class=\"figure-number\">Figure 13: </span>Superpixels optimized for semantic segmentation (Jampani et al.)" >}}

Superpixel Sampling Networks, proposed by Jampani et al., introduce the first attempt at integrating superpixel extraction methods with deep learning models (<a href="#citeproc_bib_item_3">Jampani et al. 2018</a>).
In this work, they adapt SLIC as a differentiable layer in a deep network which result in superpixels that are fine-tuned for specific tasks.

{{< figure src="/ox-hugo/2022-03-03_21-45-09_screenshot.png" caption="<span class=\"figure-number\">Figure 14: </span>Model diagram for SSN (Jampani et al.)" >}}

The train their model on a semantic segmentation task which fine tunes the learned superpixels such that they adhere more closely to segmentation boundaries.

{{< figure src="/ox-hugo/2022-03-03_21-51-28_screenshot.png" caption="<span class=\"figure-number\">Figure 15: </span>Results on semantic segmentation (Jampani et al.)" >}}

In a more recent work, Yang et al. propose a deep network that directly produces the superpixels as opposed to using a soft K-Means layer (<a href="#citeproc_bib_item_4">Yang et al. 2020</a>).

{{< figure src="/ox-hugo/2022-03-03_22-05-40_screenshot.png" caption="<span class=\"figure-number\">Figure 16: </span>Model comparison between Jampani et al. and Yang et al. (Yang et al.)" >}}

Similar to SSN, they experiment on the Berkeley Image Segmentation Dataset.
Their results are competitive with other deep learning-based approaches.
The authors note that their method generalizes better in segmentation tasks by being robust to fine details and noise.
Additionally, their model runs at 50 fps using 4 NVIDIA Titan Xp GPUs.

{{< figure src="/ox-hugo/2022-03-03_22-14-22_screenshot.png" caption="<span class=\"figure-number\">Figure 17: </span>Comparison of results on competing methods (Yang et al.)" >}}

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Achanta, Radhakrishna, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk. 2012. “SLIC Superpixels Compared to State-of-the-Art Superpixel Methods.” <i>Ieee Transactions on Pattern Analysis and Machine Intelligence</i> 34 (11): 2274–82. <a href="https://doi.org/10.1109/TPAMI.2012.120">https://doi.org/10.1109/TPAMI.2012.120</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Comaniciu, D., and P. Meer. 2002. “Mean Shift: A Robust Approach toward Feature Space Analysis.” <i>Ieee Transactions on Pattern Analysis and Machine Intelligence</i> 24 (5): 603–19. <a href="https://doi.org/10.1109/34.1000236">https://doi.org/10.1109/34.1000236</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_3"></a>Jampani, Varun, Deqing Sun, Ming-Yu Liu, Ming-Hsuan Yang, and Kautz Jan. 2018. “Superpixel Sampling Networks.” <i>Cvf European Conference on Computer Vision</i>, 17.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_4"></a>Yang, Fengting, Qian Sun, Hailin Jin, and Zihan Zhou. 2020. “Superpixel Segmentation With Fully Convolutional Networks.” <i>Computer Vision and Pattern Recognition (Cvpr)</i>, 10.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_5"></a>Zhao, Yongheng, Tolga Birdal, Haowen Deng, and Federico Tombari. 2019. “3D Point Capsule Networks.” In <i>2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 1009–18. Long Beach, CA, USA: IEEE. <a href="https://doi.org/10.1109/CVPR.2019.00110">https://doi.org/10.1109/CVPR.2019.00110</a>.</div>
</div>
