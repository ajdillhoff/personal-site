+++
title = "Object Detection"
authors = ["Alex Dillhoff"]
date = 2022-04-18T00:00:00-05:00
tags = ["computer vision", "machine learning"]
draft = false
lastmod = 2024-07-27
sections = "Computer Vision"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Papers](#papers)
- [Evaluating Object Detection Methods](#evaluating-object-detection-methods)
- [Datasets](#datasets)
- [An Incomplete History of Deep-Learning-based Object Detection](#an-incomplete-history-of-deep-learning-based-object-detection)

</div>
<!--endtoc-->



## Papers {#papers}

-   <https://awesomeopensource.com/projects/object-detection>


## Evaluating Object Detection Methods {#evaluating-object-detection-methods}

Object detection algorithms are evaluated using the mean of Average Precision (mAP) across all classes in the dataset.

Precision and recall are computed from the predictions and the ground truth. A sample and the model's prediction can either be positive or negative when it comes to classification. Either it belongs to a class or it does not. The table below summarizes the outcomes between the model's prediction and the true underlying class.

![](/ox-hugo/2022-04-17_18-17-14_screenshot.png)
Object detection algorithms are evaluated using the mean of Average Precision (mAP) across all classes in the dataset.

Precision and recall are computed from the predictions and the ground truth. A sample and the model's prediction can either be positive or negative when it comes to classification. Either it belongs to a class or it does not. The table below summarizes the outcomes between the model's prediction and the true underlying class.

**Precision**

\\[
\frac{TP}{TP + FP}
\\]

**Recall**

\\[
\frac{TP}{TP + FN}
\\]

Object detection models predict a bounding box for a given class. A correct bounding box can be identified as one that has an Intersection-over-Union (IoU) score of &gt; 0.5.

{{< figure src="/ox-hugo/2024-07-27_16-24-38_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>IoU Visualization (<a href=\"#citeproc_bib_item_7\">Szeliski 2021</a>)" >}}

If you were to plot the precision versus recall of a single class, the area under the curve would be the average precision.

{{< figure src="/ox-hugo/2024-07-27_16-32-40_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Average Precision Curves (Girshick, 2020)" >}}

The curve implicitly represents varying probability thresholds! As recall increases, precision will generally decrease. This reflects the fact that a model that recalls all input samples as a particular class will sure be misclassifying them. **Keep in mind that recall by itself is not a measure of correctness.** Ideally, the curve will be closer to the top right of the graph, indicating high precision and recall.


## Datasets {#datasets}

-   [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
-   [COCO](https://cocodataset.org/#home)


## An Incomplete History of Deep-Learning-based Object Detection {#an-incomplete-history-of-deep-learning-based-object-detection}


### Rich Feature Hierarchies for Accuracy Object Detection and Semantic Segmentation (<a href="#citeproc_bib_item_2">Girshick et al. 2014</a>) {#rich-feature-hierarchies-for-accuracy-object-detection-and-semantic-segmentation}

> The last decade of progress on various visual recognition tasks has been based considerably on the use of SIFT and HOG (<a href="#citeproc_bib_item_2">Girshick et al. 2014</a>).

This is one of the earliest papers to leverage deep learning for object detection. The overall approach is a piecewise one, where the CNN is only used for classification.

{{< figure src="/ox-hugo/2024-07-27_17-11-41_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>System overview of the R-CNN approach (<a href=\"#citeproc_bib_item_2\">Girshick et al. 2014</a>)" >}}

**Key Insights**

-   Increased mean average precision (mAP) by more than 30% on VOC 2012.
-   Candidate regions are generated using Selective Search (<a href="#citeproc_bib_item_9">Uijlings et al. 2013</a>).
-   CNNs are used to perform object classification for each region proposal.
-   Employs bounding box regression to refine the predicted bounding boxes.
-   "...when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost."


#### Region Proposals {#region-proposals}

The Selective Search algorithm is used to generate region proposals. The algorithm is based on hierarchical grouping of superpixels. Given an input image, approximately 2000 region proposals are generated. Each region proposal is a rectangular bounding box that encloses a region of interest. Since the bounding boxes are not square, the prposals are warped to a fixed size before being fed into the CNN.

{{< figure src="/ox-hugo/2024-07-27_17-17-19_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>Selective Search region proposals (<a href=\"#citeproc_bib_item_9\">Uijlings et al. 2013</a>)." >}}


#### Feature Extraction {#feature-extraction}

Each warped image is passed through a CNN backbone pre-trained on ImageNet. The output is a 4096-dimensional feature vector. In the context of today's methods, this certainly does not seem like a very sophisticated approach. Given the context of the time, this was a significant improvement over the SIFT and HOG features that were previously used. The other benefit is that the CNN can be fine-tuned on the target dataset, if desired.


#### Classification {#classification}

The feature vectors from the CNN are used as input to a linear SVM for classification. The SVM is trained to classify the region proposals into one of the classes in the dataset. As stated before, this really is a piecewise solution. It is also very slow, as each region proposal must be passed through the CNN and SVM.


#### Training {#training}

The CNN and SVM both need to be trained. The CNN is first pre-trained on ImageNet, but it still needs to be fine-tuned on the target dataset. Not only is the target dataset out of domain, but the images themselves are warped. The authors do admit that it is reasonable to simply use the output of the softmax layer for classification in stead of training an SVM. In their experiments, they found that the SVM provided a slight improvement in performance.


#### Inference {#inference}

Given roughly 2000 region proposals, how is a final prediction made? For each class, non-maximum suppression is applied to the region proposals. A region is rejected if it has an IoU overlap with a higher scoring region above a certain threshold.


### Fast R-CNN (<a href="#citeproc_bib_item_1">Girshick 2015</a>) {#fast-r-cnn}

Published roughly a year after the original R-CNN paper, Fast R-CNN addresses many of the shortcomings and ineffeciencies of the original approach. The main innovation is the introduction of the Region of Interest (RoI) pooling layer. This layer allows the CNN to be applied to the entire image, rather than to each region proposal individually.

{{< figure src="/ox-hugo/2024-07-27_17-55-06_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>System overview of Fast R-CNN (<a href=\"#citeproc_bib_item_1\">Girshick 2015</a>)" >}}

**Key Insights**

-   Single-stage training joinly optimizes a softmax classifier and bounding box regressor.
-   RoI pooling layer allows the CNN to be applied to the entire image.
-   Object proposals are still provided by Selective Search, but only a single forward pass is needed to extract features and make predictions.


#### RoI Pooling {#roi-pooling}

Given an input image, a feature extracting CNN produces a feature map. For each region proposal the RoI pooling layer extracts a feature vector directly from the feature map. The feature vector is passed through a fully connected network to produce a class score and bounding box regression.

A **region proposal** is defined by a four-tuple \\((r, c, h, w)\\) defining the top-left corner \\((r, c)\\) and the height and width \\((h, w)\\) of the region. The RoI pooling layer divides the window into a \\(H \times W\\) grid of sub-windows, where \\(H\\) and \\(W\\) are hyperparameters. A max pooling operation is applied to each sub-window to produce a fixed-size output.


#### Training {#training}

Three different CNN backbones were tested. The last max pooling layer is replaced with the RoI pooling layer. From there, the two branches of the network are added: a softmax classifier and a bounding box regressor. The softmax classifier is trained to classify the region proposals into one of the classes in the dataset. The bounding box regressor is trained to refine the predicted bounding boxes.

Another key efficiency improvement comes from hierarchical sampling during training. Since the RoI pooling layer allows the CNN to be applied to the entire image, the same feature map can be used for multiple region proposals. This is in contrast to the original R-CNN approach, where each region proposal was passed through the CNN individually. When training, the authors sample a mini-batch of \\(N\\) images and \\(R\\) regions, yielding \\(R / N\\) RoIs from each image.


#### Results {#results}

This work achieved state-of-the-art results on VOC 2007 (70.0), 2010 (68.8), and 2012 (68.4). Additionally, the model was able to run at about 5 frames per second on a GPU.


### Faster R-CNN (<a href="#citeproc_bib_item_6">Ren et al. 2017</a>) {#faster-r-cnn}

Fast R-CNN improved on many of the glaring issues presented in the original R-CNN paper. One major bottleneck left was the Selective Search algorithm used to generate region proposals. In the third iteration, Faster R-CNN, the authors introduce the Region Proposal Network (RPN) to replace slower region proposal methods.

**Key Insights**

-   Region Proposal Network (RPN) generates region proposals.
-   RPN is a fully convolutional network that shares features with the object detection network.
-   Anchor boxes of varying aspect ratios are used to predict region proposals.


#### Region Proposal Network {#region-proposal-network}

The RPN uses a sliding window approach to generate region proposals. Instead of using a separate CNN, it leverages the feature maps generated by a backbone CNN such as VGG or ResNet.

{{< figure src="/ox-hugo/2024-07-28_17-59-11_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>Region Proposal Network (RPN) (<a href=\"#citeproc_bib_item_6\">Ren et al. 2017</a>)" >}}

The RPN produces a feature vector for each sliding window, which is fed into a bounding box regressor and classifier. For each of these sliding windows, \\(k\\) **anchor** boxes are generated. In their experiments, they generate 9 anchors based on 3 scales and 3 aspect ratios.

{{< figure src="/ox-hugo/2024-07-28_18-02-38_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>Anchor boxes (<a href=\"#citeproc_bib_item_6\">Ren et al. 2017</a>)" >}}

**Why use anchor boxes instead of directly predicting bounding boxes?** The authors argue that using anchor boxes simplifies the problem. It adds translation invariance: if an object is translated, the proposal should translate accordingly. This approach is also more cost-efficient. Consider an input image for which multiple scales of bounding boxes are generated. To generate multiple scales, the image would need to be resized and passed through the network multiple times. With anchor boxes, the network only needs to be run once.

{{< figure src="/ox-hugo/2024-07-28_18-12-19_screenshot.png" caption="<span class=\"figure-number\">Figure 9: </span>Different schemes for addressing multiple scales and sizes (<a href=\"#citeproc_bib_item_6\">Ren et al. 2017</a>)." >}}


#### Results {#results}

Faster R-CNN achieved state-of-the-art results on VOC 2007 (73.2) and 2012 (70.4). These scores increase to 78.8 and 75.9 when using the COCO dataset. Additionally, they evaluate on MS COCO, achieving a mAP@.5 of 42.7, meaning that the model is able to detect 42.7% of the objects in the dataset with an IoU of 0.5 or greater. This score goes down to 21.9% when the IoU threshold is expanded in the range of 0.5 to 0.95.


### You Only Look Once {#you-only-look-once}

You Only Look Once (YOLO) is a single-stage object detection algorithm that is able to predict multiple bounding boxes and class probabilities in a single forward pass (<a href="#citeproc_bib_item_5">Redmon et al. 2016</a>). Since 2016, it has benefitted from many improvements, some documented by peer-reviewed papers and others by the community. For a recent survey of YOLO and its variants, see (<a href="#citeproc_bib_item_8">Terven, Córdova-Esparza, and Romero-González 2023</a>).


#### YOLOv1 (<a href="#citeproc_bib_item_5">Redmon et al. 2016</a>) {#yolov1}

{{< figure src="/ox-hugo/2024-07-28_20-02-46_screenshot.png" caption="<span class=\"figure-number\">Figure 10: </span>YOLOv1 Overview (<a href=\"#citeproc_bib_item_5\">Redmon et al. 2016</a>)" >}}

The original YOLO method works by dividing an input image into a \\(S \times S\\) grid. Each grid cell predicts \\(B\\) bounding boxes and confidence scores for each box, along with \\(C\\) class probabilities.

{{< figure src="/ox-hugo/2024-07-28_20-18-23_screenshot.png" caption="<span class=\"figure-number\">Figure 11: </span>The Model (<a href=\"#citeproc_bib_item_5\">Redmon et al. 2016</a>)." >}}

During training, only one predictor should be responsible for each object. This is enforced by assigning a predictor to an object based on the highest IoU with the ground truth. For inference, the model outputs bounding boxes with a confidence score greater than a threshold. The entire model is trained using a multi-part loss function that includes terms for objectness, classification, and bounding box regression.

{{< figure src="/ox-hugo/2024-07-28_20-21-15_screenshot.png" caption="<span class=\"figure-number\">Figure 12: </span>Annotated description of YOLO loss (<a href=\"#citeproc_bib_item_8\">Terven, Córdova-Esparza, and Romero-González 2023</a>)." >}}

YOLOv1 was evaluated on the VOC 2007 dataset, achieving a mAP of 63.4. The model was able to run at 45 frames per second on a GPU.


#### YOLOv2: Better, Faster, and Stronger (<a href="#citeproc_bib_item_3">Redmon and Farhadi 2016</a>) {#yolov2-better-faster-and-stronger}

YOLOv2 was publised at CVPR 2017 and introduced several key improvements over YOLOv1.

-   **Batch normalization** acts as a regularizer to reduce overfitting during training.
-   **Fully convolutional**. All dense layers are replaced with convolutional layers.
-   Following Faster R-CNN, YOLOv2 uses **anchor boxes** to predict bounding boxes.
-   A predetermined set of anchor box sizes are computed using k-means clustering on the training data.
-   The model was trained to be robust to varying sizes via **multi-scale training**, where the input image is randomly resized during training.
-   **Multi-task Learning**. The network is pre-trained on ImageNet, where no object detection labels are used. In the event that an input sample does not contain a bounding box annotation, the model is trained to predict the background class.

YOLOv2 achieved a 73.4% mAP on VOC 2012, beating Faster R-CNN.


#### YOLOv3 (<a href="#citeproc_bib_item_4">Redmon and Farhadi 2018</a>) {#yolov3}

The third improvement to YOLO was not published at a major conference, but was released on arXiv. The main improvement is a deeper 53-layer backbone network with residual connections. The new method also supports multi-scale predictions by predicting three boxes at three different scales. The same anchor box computation via k-means is done in this work, with the number of priors being expanded to support three different scales.


#### YOLOv4 and beyond {#yolov4-and-beyond}

The remaining iterations were developed by other members of the community and introduce several key improvements while maintaining the spirit and goals of the original paper. For more information, see the informative survey paper by (<a href="#citeproc_bib_item_8">Terven, Córdova-Esparza, and Romero-González 2023</a>).

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Girshick, Ross. 2015. “Fast R-CNN,” April. <a href="https://doi.org/10.48550/arXiv.1504.08083">https://doi.org/10.48550/arXiv.1504.08083</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Girshick, Ross, Jeff Donahue, Trevor Darrell, and Jitendra Malik. 2014. “Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation.” <i>Arxiv:1311.2524 [Cs]</i>, October. <a href="http://arxiv.org/abs/1311.2524">http://arxiv.org/abs/1311.2524</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_3"></a>Redmon, Joseph, and Ali Farhadi. 2016. “YOLO9000: Better, Faster, Stronger.” <i>Arxiv:1612.08242 [Cs]</i>, December. <a href="http://arxiv.org/abs/1612.08242">http://arxiv.org/abs/1612.08242</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_4"></a>———. 2018. “YOLOv3: An Incremental Improvement,” 6.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_5"></a>Redmon, Joseph, Santosh Divvala, Ross Girshick, and Ali Farhadi. 2016. “You Only Look Once: Unified, Real-Time Object Detection.” <i>Arxiv:1506.02640 [Cs]</i>, May. <a href="http://arxiv.org/abs/1506.02640">http://arxiv.org/abs/1506.02640</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_6"></a>Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun. 2017. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” <i>Ieee Transactions on Pattern Analysis and Machine Intelligence</i> 39 (6): 1137–49. <a href="https://doi.org/10.1109/TPAMI.2016.2577031">https://doi.org/10.1109/TPAMI.2016.2577031</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_7"></a>Szeliski, Richard. 2021. <i>Computer Vision: Algorithms and Applications</i>. 2nd ed. <a href="http://szeliski.org/Book/2ndEdition.htm">http://szeliski.org/Book/2ndEdition.htm</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_8"></a>Terven, Juan, Diana-Margarita Córdova-Esparza, and Julio-Alejandro Romero-González. 2023. “A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS.” <i>Machine Learning and Knowledge Extraction</i> 5 (4): 1680–1716. <a href="https://doi.org/10.3390/make5040083">https://doi.org/10.3390/make5040083</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_9"></a>Uijlings, J. R. R., K. E. A. van de Sande, T. Gevers, and A. W. M. Smeulders. 2013. “Selective Search for Object Recognition.” <i>International Journal of Computer Vision</i> 104 (2): 154–71. <a href="https://doi.org/10.1007/s11263-013-0620-5">https://doi.org/10.1007/s11263-013-0620-5</a>.</div>
</div>
