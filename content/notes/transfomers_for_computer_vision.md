+++
title = "Transformers for Computer Vision"
authors = ["Alex Dillhoff"]
date = 2023-04-18T00:00:00-05:00
tags = ["computer vision", "machine learning"]
draft = false
lastmod = 2025-02-19
sections = "Computer vision"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Vision Transformer (ViT) (<a href="#citeproc_bib_item_2">Dosovitskiy et al. 2021</a>)](#vision-transformer--vit)
- [Swin Transformer (<a href="#citeproc_bib_item_4">Liu et al. 2021</a>)](#swin-transformer)

</div>
<!--endtoc-->



## Vision Transformer (ViT) (<a href="#citeproc_bib_item_2">Dosovitskiy et al. 2021</a>) {#vision-transformer--vit}

The original Vision Transformer (ViT) was published by Google Brain with a simple objective: apply the Transformer architecture to images, adding as few modifications necessary. When trained on ImageNet, as was standard practice, the performance of ViT does not match models like ResNet. However, scaling up to hundreds of millions results in a better performing model.

This point is highlighted in the introduction of the paper, but isn't that something that should be expected? If anything, that result simply implies that CNNs are more parameter efficient than ViT. Seeing as this was one of the first adaptations of this architecture to vision, some slack should be given.


### The Model {#the-model}

{{< figure src="/ox-hugo/2024-07-30_17-12-54_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Overview of ViT (<a href=\"#citeproc_bib_item_2\">Dosovitskiy et al. 2021</a>)." >}}

The first challenge is to figure out how to pass an image as input to a Transformer. In the original paper, the input is a sequence of text embeddings. If we get creative, there are a number of ways we could break up an image.

1.  Split the image into individual patches.

    This is what the ViT paper does. Each patch is flattened to a \\(P^2 \cdot C\\) vector, where \\(P\\) is the size of each square patch. Another reason the image is split into patches is to reduce the size of weight matrix of the attention layers. Using the entire image as a single input would be prohibitive.

2.  Process individual image patches through a CNN.

    This approach has been tried with some success, but isn't the preferred approach for this paper. The goal, after all, is to conform as closely as possible to the original architecture. One could really run away with this idea. Processing each patch independently through a CNN woud be cumbersome and most likely result in a loss of spatial information. The original image could be processed through a CNN before splitting up the feature maps. This is discussed in the ViT paper as the **hybrid approach**.

3.  Use output feature maps from a CNN.

    This approach relates somewhat to number 2. Given a series of output feature maps \\(X \in \mathbb{R}^{C \times H \times W}\\), pass each feature map as its own _patch_ a la the author's approach. The feature maps would not be related spatially, but through the channels.

Taking the first approach, the patches are projected to a \\(D\\) dimensional vector the authors refer to as **patch embeddings**.


#### Learnable Embeddings {#learnable-embeddings}

As is done in BERT, the author's attach a `[class]` token to the sequence of embedded patches. The purpose of this is to aggregate information from all the patches via self-attention. The output is a learned global representation of the input image. This output vector is then passed through a classification head.


#### Position Embeddings {#position-embeddings}

Two immediate choices for position embeddings are either 1D or 2D-aware. The embeddings themselves are learnable. The authors opt for 1D learnable embeddings due to their simplicity and the fact that there was no noticeable performance difference when using 2D-aware embeddings.

The position embeddings are crucial for ViT since the each patch is represented as an embedding. While each patch embedding does retain locality, the relationship between the patches is not preserved. This relationship is learned via position embeddings. Basically, the **inductive biases** that CNNs have must be learned in a ViT.


#### Using Higher Resolutions {#using-higher-resolutions}

When using higher resolution input, the patch size is kept the same. This means that the sequence length is increases to accommodate the larger number of patches. Although the input to the Transformer can handle an abitrary sequence length, the position embeddings learned at one resolution are useless when fine-tuned on another.


### Evaluations {#evaluations}

Most of the information in the original ViT paper is dedicated to experiments. The model itself is a simple adaptation. The primary goal then is to evaluate its characteristics and how it performs compared to CNNs.

**Key Results**

-   ViT achieves better downstream accuracy than SOTA ResNet-based approaches using JFT-300M for pre-training.
-   ViT requires fewer computational resources.
-   CNN models perform better when pre-trained on smaller datasets compared to ViT, but ViT surpasses them when using very large datasets.


### Properties {#properties}

The principal components of the patch embeddings resemble features learned in the early layers of a CNN. The authors did not investigate whether ViT builds a hierarchy of features similar to CNNs. However, later work confirmed that the features learned at each layer of a ViT progress in a similar manner (<a href="#citeproc_bib_item_3">Ghiasi et al. 2022</a>).

{{< figure src="/ox-hugo/2024-07-31_15-29-26_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>The progression for visualized features of ViT B-32 (<a href=\"#citeproc_bib_item_3\">Ghiasi et al. 2022</a>)." >}}

The position embeddings were also analyzed and the authors found that patches that are grouped together tend to have similar embeddings. The fact that these embeddings learned 2D relationships gives weight to the argument that hand-crafted 2D-aware embeddings do not perform any better.

{{< figure src="/ox-hugo/2024-07-31_15-37-18_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Position embedding similarity (<a href=\"#citeproc_bib_item_2\">Dosovitskiy et al. 2021</a>)." >}}

The self-attention mechanisms in ViT attend to information globally across the entire image. The authors verify this by computing the average distance in image space across which the information is used. The distance varies between the different attention heads. Some heads attend to most of the detail in the lowest layer. This suggests that global information is used by the model since it attends across the representations in these early layers. Other heads have a smaller attention distance, suggesting they are focusing on local information.

One interesting result with respect to the hybrid models it that they do not observe as much local attention. This is most likely because local spatial relationships are captured by the early layers of a CNN. In hybrid models, the Transformer received processed feature maps from CNNs.


### Self-Supervision {#self-supervision}

Self-supervised pre-training is one of the key ingredients to training LLMs (<a href="#citeproc_bib_item_5">Radford et al. 2018</a>), (<a href="#citeproc_bib_item_1">Devlin et al. 2019</a>). The authors of ViT experiment with self-supervised pre-training by masking out different patches and having the model predict the masked regions, similar to BERT (<a href="#citeproc_bib_item_1">Devlin et al. 2019</a>). With self-supervised pre-training, ViT-B/16 achieves 79.9% accuracy on ImageNet, 2% higher compared to training from scratch.


## Swin Transformer (<a href="#citeproc_bib_item_4">Liu et al. 2021</a>) {#swin-transformer}

-   Introduces a general backbone for computer vision tasks.
-   Builds a _hierarchical_ transformer to allow for non-overlapping windows.
-   Can be used for classification, object detection, and instance segmentation.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” arXiv. <a href="https://doi.org/10.48550/arXiv.1810.04805">https://doi.org/10.48550/arXiv.1810.04805</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, et al. 2021. “An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.” <i>Arxiv:2010.11929 [Cs]</i>, June. <a href="http://arxiv.org/abs/2010.11929">http://arxiv.org/abs/2010.11929</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_3"></a>Ghiasi, Amin, Hamid Kazemi, Eitan Borgnia, Steven Reich, Manli Shu, Micah Goldblum, Andrew Gordon Wilson, and Tom Goldstein. 2022. “What Do Vision Transformers Learn? A Visual Exploration.” arXiv. <a href="https://doi.org/10.48550/arXiv.2212.06727">https://doi.org/10.48550/arXiv.2212.06727</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_4"></a>Liu, Ze, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. 2021. “Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows.” arXiv. <a href="https://doi.org/10.48550/arXiv.2103.14030">https://doi.org/10.48550/arXiv.2103.14030</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_5"></a>Radford, Alec, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. “Improving Language Understanding by Generative Pre-Training,” 12.</div>
</div>
