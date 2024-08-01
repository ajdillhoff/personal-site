+++
title = "Instance Segmentation"
authors = ["Alex Dillhoff"]
date = 2022-04-18T00:00:00-05:00
tags = ["computer vision", "machine learning"]
draft = false
lastmod = 2024-07-29
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Mask R-CNN (<a href="#citeproc_bib_item_4">He et al. 2018</a>)](#mask-r-cnn)
- [CenterMask (<a href="#citeproc_bib_item_6">Lee and Park 2020</a>)](#centermask)
- [Cascade R-CNN (<a href="#citeproc_bib_item_1">Cai and Vasconcelos 2019</a>)](#cascade-r-cnn)
- [MaskFormer (<a href="#citeproc_bib_item_3">Cheng, Schwing, and Kirillov 2021</a>)](#maskformer)
- [Mask2Former (<a href="#citeproc_bib_item_2">Cheng et al. 2022</a>)](#mask2former)
- [Mask-FrozenDETR (<a href="#citeproc_bib_item_7">Liang and Yuan 2023</a>)](#mask-frozendetr)
- [Segment Anything (<a href="#citeproc_bib_item_5">Kirillov et al. 2023</a>)](#segment-anything)
- [Segment Anything 2 (<a href="#citeproc_bib_item_9">Ravi et al. 2024</a>)](#segment-anything-2)

</div>
<!--endtoc-->



## Introduction {#introduction}


## Mask R-CNN (<a href="#citeproc_bib_item_4">He et al. 2018</a>) {#mask-r-cnn}

Mask R-CNN adapts Faster R-CNN to include a branch for instance segmentation (<a href="#citeproc_bib_item_10">Ren et al. 2017</a>). This branch predicts a binary mask for each RoI, and the training loss is updated to include this branch.

{{< figure src="/ox-hugo/2024-07-29_14-03-43_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Mask R-CNN framework (<a href=\"#citeproc_bib_item_4\">He et al. 2018</a>)." >}}

**Key Contributions**

-   Introduces RoIAlign to preserve exact spatial locations.
-   Decouples mask and class predictions, allowing the network to generate masks for each class without competition among classes.
-   Achieved SOTA reults on COCO instance segmentation, object detection, and human pose estimation.


### RoIAlign {#roialign}

Transforming a spatial-preserving representation into a compressed output necessarily removes that spatial encoding. In other words, the spatial information is lost when the feature map is downsampled. To address this, Mask R-CNN introduces RoIAlign, which preserves the spatial information of the feature map. RoIAlign is a bilinear interpolation method that samples the feature map at the exact locations of the RoI.

Regions of Interest (RoIs) are generated based on the output feature map of the backbone network. These bounding boxes do not line up perfectly with the feature map. RoIPooling would round the coordinates to the nearest integer, which can lead to misalignment. This was not an issue in Faster R-CNN, where the goal was to predict the class and bounding box. However, for instance segmentation, the spatial information is crucial.

{{< figure src="/ox-hugo/2024-07-29_14-10-35_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>RoIAlign. Dashed grid - feature map, solid grid - RoI (<a href=\"#citeproc_bib_item_4\">He et al. 2018</a>)." >}}

As seen in the figure above, RoIAlign computes the exact values of the feature map at regularly sampled locations in teach RoI bin. The sampled points in each bin are aggregated via max or average pooling. The output is still the same size as the RoI, but it takes data from the feature map at the exact locations of the RoI.


### Mask Head {#mask-head}

Given the feature map produce by RoIAlign, the mask network head is a small convolutional network that upsamples the feature map using a series of convolutions and deconvolutions. The output is a binary mask for each RoI. Not only does the mask head serve to decouple the mask prediction from box and class prediction, but it also allows the network to learn features specific to mask generation.

{{< figure src="/ox-hugo/2024-07-29_14-38-38_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Head Architecture (<a href=\"#citeproc_bib_item_4\">He et al. 2018</a>)." >}}


### Results {#results}

Mask R-CNN achieved state-of-the-art results on the COCO dataset for instance segmentation, object detection, and human pose estimation. Using ResNeXt-101-FPN, Mask R-CNN achieved better performance over the leading models from the previous year's competition, netting 37.1% AP on the COCO test-dev set.

{{< figure src="/ox-hugo/2024-07-29_18-26-12_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Qualitative results versus FCIS (leading competitor) (<a href=\"#citeproc_bib_item_4\">He et al. 2018</a>)." >}}


## CenterMask (<a href="#citeproc_bib_item_6">Lee and Park 2020</a>) {#centermask}

CenterMask is a real-time anchor-free instance segmentation method. It adds a novel Spatial Attention-Guided Mask branch on top of the FCOS object detector.

**Summary**

-   Anchor-free approach for bounding boxes.
-   Spatial Attention-Guided Mask branch for instance segmentation.
-   Two-stage architecture for object detection and mask prediction.
-   Objects are represented by their center key points and bounding box sizes.
-   Outperforms Mask R-CNN on common benchmarks.

{{< figure src="/ox-hugo/2024-07-29_18-39-14_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>Architecture of CenterMask (<a href=\"#citeproc_bib_item_6\">Lee and Park 2020</a>)." >}}


### Backbone {#backbone}

The authors adapt VoVNet (Variety of View), a CNN-based architecture, as the backbone network. VoVNet is designed to capture multi-scale features and has a high computational efficiency. VoVNet2 adds a spatial attention module to the original VoVNet architecture along with residual connections.

{{< figure src="/ox-hugo/2024-07-29_18-49-52_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>VoVNet2 Backbone comparison (<a href=\"#citeproc_bib_item_6\">Lee and Park 2020</a>)." >}}

The residual connections were added to the original VoVNet architecture to improve the gradient flow during training. In the original network, performance degradation occurred when stacking multile One-Shot Aggregation (OSA) blocks. The OSA blocks are designed to capture multi-scale features by aggregating information from different scales via successive convolutions. The features are concatenated and passed through a bottleneck layer to reduce the number of channels.

The effective Squeeze-Excitation layer takes the concatenated features and applies a global average pooling operation per channel. The produces a \\(1 \times 1 \times C\\) tensor, where \\(C\\) is the number of channels. This is passed through a fully connected layer with a sigmoid function to produce a channel-wise attention map. The attention map is then multiplied with the input features to produce the final output. This allows the network to focus on the most important features.


### Feature Pyramid Networks (<a href="#citeproc_bib_item_8">Lin et al. 2017</a>) {#feature-pyramid-networks}

The output of the backbone network is passed through a Feature Pyramid Network (FPN) to extract multi-scale features. The FPN is a top-down architecture with lateral connections that allow the network to capture features at different scales. Since multiple scales are produced, RoIAlign must be adapted to handle them.

{{< figure src="/ox-hugo/2024-07-29_19-52-54_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>FPN Architecture (<a href=\"#citeproc_bib_item_8\">Lin et al. 2017</a>)." >}}


### Scale-adaptive RoI Assignment {#scale-adaptive-roi-assignment}

The output of the backbone + FPN is a series of feature maps at different scales. The authors propose a scale-adaptive RoI assignment method to assign RoIs to the appropriate feature map. This is done by calculating the area of the RoI and assigning it to the feature map that best matches the RoI size. This allows the network to focus on the most relevant features for each object. Given the appropriate feature map, RoIAlign is used to extract the features for each RoI.


### FCOS Detection Head {#fcos-detection-head}

FCOS is a region proposal network that predicts bounding boxes and class labels without using predefined anchor boxes. Object detection is treated as a dense, per-pixel prediction task. The predicted bounding boxes are used to crop the feature map, which is then passed to the mask branch.

**Summary**

-   Uses a CNN with a Feature Pyramid Network to extract multi-scale features. In this case, the VoVNet2 backbone is used.
-   Predicts a 4D vector plus a class label at each spatial location of a feature map.
-   Predicts deviation of a pixel from the center of its bounding box.


### SAG-Mask Branch {#sag-mask-branch}

The Spatial Attention-Guided Mask branch highlights meaningful pixels while suppressing irrelevant ones via spatial attention. The input to this branch are the features extracted from RoI Align. These come from the backbone + FPN module.

The feature maps go through a round of 4 convolutional layers for further feature processing. The SAM module itself generates average and max-pooled features along the channel axis: \\(P\_{avg},\ P\_{max} \in \mathbb{R}^{1 \times W \times H}\\). These are concatenated and processed through another convolutional layer with a sigmoid activation function.

\\[
A\_{sag}(X\_i) = \sigma(F\_{3 \times 3}(P\_{max} \circ P\_{avg}))
\\]

The sigmoid ensures that each output value represents how much attention should be paid to the original input features. This is then multiplied element-wise with the original output, resuling in an attention-guided feature map.

\\[
X\_{sag} = A\_{sag}(X\_i) \otimes X\_i
\\]


### Results {#results}

CenterMask achieved 40.6% mask AP (over all thresholds) using their base model, a 1.3% improvement over Mask R-CNN. They also achieved 41.8% when built using Detectron2. However, Detectron2 was released after their original submission, so the results are not official.


## Cascade R-CNN (<a href="#citeproc_bib_item_1">Cai and Vasconcelos 2019</a>) {#cascade-r-cnn}

This paper addresses two key problems with high-quality detections.

1.  Overfitting due to vanishing positive samples for large thresholds.

    When training object detectors, the IoU threshold is used to determine whether a detection is a positive or negative sample. A higher IoU is a stricter criteria for what constitutes a positive sample. As this threshold increases, the number of positive samples decreases since it is more challenging for the model to detect. **Typically, the proposals are selected as positive examples during training if they have at least 0.5 IoU. Raising that threshold means that the model sees fewer positive samples, leading to overfitting.**

2.  Inference-time quality mismatch between detector and test hypotheses.

    Training deals with lower quality samples (IoU ~0.5), but test samples include a range of low and high quality samples. This can lead to poor performance during inference.

**Summary**

-   Uses an RPN as in Faster R-CNN.
-   Performs iterative refinements to bounding box predictions using multi-stage detection.
-   Achieves SOTA performance for object detection (50.9% AP) and instance segmentation (42.3% AP).

{{< figure src="/ox-hugo/2024-07-30_09-21-27_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>Comparison of architectures (<a href=\"#citeproc_bib_item_1\">Cai and Vasconcelos 2019</a>)." >}}


### Multi-Stage Detection Heads {#multi-stage-detection-heads}

Multi-stage detection allows the model to progressively refine bounding box predictions. This allows the model to train with both low and high quality samples, alleviating the problems of overfitting and quality mismatch.


#### Initial Detection {#initial-detection}

The method begins similarly to Faster R-CNN. A RPN generates the initial proposals which are processed by the first detection stage. The output is a set of refined bounding boxes along with object labels and confidence scores. The first stage follows previous convention and sets the IoU of positive samples to 0.5. That is, positive samples are labeled as such in training if they have &gt;= 0.5 IoU with the ground truth.


#### Progressive Refinement {#progressive-refinement}

Each stage after that uses the output bounding boxes from the previous stage as input. The IoU is increased during training in subsequent stages to focus on higher quality detections. For example, if stage one trains on 0.5 IoU, then stage two will take those bounding boxes and select positive samples with 0.6 IoU, and so on.


### Cascade Mask R-CNN {#cascade-mask-r-cnn}

{{< figure src="/ox-hugo/2024-07-30_09-15-23_screenshot.png" caption="<span class=\"figure-number\">Figure 9: </span>Original Mask R-CNN (left) and three different Cascade Mask R-CNN strategies (right) (<a href=\"#citeproc_bib_item_1\">Cai and Vasconcelos 2019</a>)." >}}

To adapt the model for instance segmentation, the segmentation branch is inserted in parallel to the detection branch. The main question during development was _where_ to add the segmentation branch and _how many_ should be added. The authors try several different configurations, as seen in the figure above. Experimentally, the third (right-most) strategy depicted above yields the greatest performance of 35.5% AP.


## MaskFormer (<a href="#citeproc_bib_item_3">Cheng, Schwing, and Kirillov 2021</a>) {#maskformer}


## Mask2Former (<a href="#citeproc_bib_item_2">Cheng et al. 2022</a>) {#mask2former}


## Mask-FrozenDETR (<a href="#citeproc_bib_item_7">Liang and Yuan 2023</a>) {#mask-frozendetr}


## Segment Anything (<a href="#citeproc_bib_item_5">Kirillov et al. 2023</a>) {#segment-anything}


## Segment Anything 2 (<a href="#citeproc_bib_item_9">Ravi et al. 2024</a>) {#segment-anything-2}

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Cai, Zhaowei, and Nuno Vasconcelos. 2019. “Cascade R-CNN: High Quality Object Detection and Instance Segmentation.” arXiv. <a href="https://doi.org/10.48550/arXiv.1906.09756">https://doi.org/10.48550/arXiv.1906.09756</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Cheng, Bowen, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. 2022. “Masked-Attention Mask Transformer for Universal Image Segmentation.” In <i>2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 1280–89. New Orleans, LA, USA: IEEE. <a href="https://doi.org/10.1109/CVPR52688.2022.00135">https://doi.org/10.1109/CVPR52688.2022.00135</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_3"></a>Cheng, Bowen, Alexander G Schwing, and Alexander Kirillov. 2021. “Per-Pixel Classiﬁcation Is Not All You Need for Semantic Segmentation.” <i>Neurips</i>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_4"></a>He, Kaiming, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. 2018. “Mask R-CNN.” <i>Arxiv:1703.06870 [Cs]</i>, January. <a href="http://arxiv.org/abs/1703.06870">http://arxiv.org/abs/1703.06870</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_5"></a>Kirillov, Alexander, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, et al. 2023. “Segment Anything.” arXiv. <a href="http://arxiv.org/abs/2304.02643">http://arxiv.org/abs/2304.02643</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_6"></a>Lee, Youngwan, and Jongyoul Park. 2020. “CenterMask : Real-Time Anchor-Free Instance Segmentation.” arXiv. <a href="https://doi.org/10.48550/arXiv.1911.06667">https://doi.org/10.48550/arXiv.1911.06667</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_7"></a>Liang, Zhanhao, and Yuhui Yuan. 2023. “Mask Frozen-DETR: High Quality Instance Segmentation with One GPU.” arXiv. <a href="http://arxiv.org/abs/2308.03747">http://arxiv.org/abs/2308.03747</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_8"></a>Lin, Tsung-Yi, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. 2017. “Feature Pyramid Networks for Object Detection.” arXiv. <a href="https://doi.org/10.48550/arXiv.1612.03144">https://doi.org/10.48550/arXiv.1612.03144</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_9"></a>Ravi, Nikhila, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, et al. 2024. “SAM 2: Segment Anything in Images and Videos,” July.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_10"></a>Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun. 2017. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” <i>Ieee Transactions on Pattern Analysis and Machine Intelligence</i> 39 (6): 1137–49. <a href="https://doi.org/10.1109/TPAMI.2016.2577031">https://doi.org/10.1109/TPAMI.2016.2577031</a>.</div>
</div>
