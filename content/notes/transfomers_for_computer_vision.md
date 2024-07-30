+++
title = "Transformers for Computer Vision"
authors = ["Alex Dillhoff"]
date = 2023-04-18T00:00:00-05:00
tags = ["computer vision", "machine learning"]
draft = false
lastmod = 2024-07-29
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Adapting Transformers for Image Classification](#adapting-transformers-for-image-classification)
- [Notable Developments](#notable-developments)

</div>
<!--endtoc-->



## Adapting Transformers for Image Classification {#adapting-transformers-for-image-classification}


## Notable Developments {#notable-developments}


### iGPT {#igpt}

-   Motivated by the success of pre-training in models like BERT and GPT-2.
-   Auto-regressively predicts pixels, meaning it is trained to reproduce its input from missing values.
-   Does not encode 2D spatial structure, but still seems to "understand" it.
-   Input is given in "raster-order": top-to-bottom, left-to-right.
-   Predicts raw pixel values.
-   Demonstrate the effectiveness of learned features by finetuning on downstream tasks such as image classification.
-   Due to context size, the input has to be scaled down to \\(32 \times 32 \times 3\\).


### ViT {#vit}

-   Divide the input into sequence of 2D patches \\(\mathbf{x}\_p \in \mathbb{R}^{N \times (P^2 \cdot C)}\\), where \\(C\\) is the number of channels, \\((P, P)\\) is the resolution of each patch, and \\(N = HW/P^2\\).
-   Embed the patches with a linear projection. The embedded vectors have \\(D\\) dimensions.
-   Use an off-the-shelf Transformer encoder.
-   Positional encoding via 1D position embeddings. 2D position embeddings are more complicated and did not provide a better result.
-   Output of encoder is input to linear classification head.
-   The full image could not be used due to the attention layers. This would require some \\(W \in \mathbb{R}^{(H \times W \times C)^2}\\)
-   Create 196 patches of \\(3 \times 16 \times 16\\) flattened to \\(196 \times 768\\).
-   A classification token is added to the image patch.
-   More parameters and training compared to CNNs.
