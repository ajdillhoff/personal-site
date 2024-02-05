+++
title = "Bag of Visual Words"
authors = ["Alex Dillhoff"]
date = 2024-02-04T18:54:00-06:00
tags = ["computer vision", "machine learning"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Bag of Visual Words](#bag-of-visual-words)

</div>
<!--endtoc-->

**Bag of Words** is a technique used in Natural Language Processing for document classification. It is a collection of word counts. To create a Bag of Words for a document, it necessary to create a dictionary first. Choosing the a dictionary is based on many factors including computational limitations. Next, the documents in a dataset are tokenized into words. The word counts are collected as part of a histogram and used as a feature vector for a machine learning model.

The dictionary is the same for all documents in the original dataset. Ideally, the Bag of Word vectors for each document in the same class will be similar. This technique works well for problems in natural language processing, where each input document will have a varying number of words. By using a Bag of Words, the input data is transformed into a fixed length feature vector.


## Bag of Visual Words {#bag-of-visual-words}

The Bag of Visual Words model adapts this technique to computer vision. Instead of words, distinct visual features are extracted from each image. Some images may have more features than others, similar to how some documents will have different word counts. The dictionary is created by clustering the visual features into a finite number of groups, determined as a hyperparameter. The visual features for each image are then counted and used as a feature vector for a machine learning model.


### Extract Visual Features {#extract-visual-features}

The first step in creating a Bag of Visual Words is to extract visual features from each image. The visual features are typically extracted using a technique like SIFT, SURF, or ORB. These techniques are designed to extract features that are invariant to scaling, rotation, and translation. The visual features are then stored in a list for each image.


### Create Visual Words {#create-visual-words}

Creating the dictionary requires clustering the features into a finite number of groups. The number of groups will vary depending on the complexity of the data. For a given dataset, this can be determined empirically. The most common clustering algorithm for this is K-Means, in which \\(k\\) different clusters are created and updated iteratively. The visual features are then assigned to the nearest cluster, and the cluster centers are updated. This process is repeated until the cluster centers converge.


### Build Sparse Frequency Vectors {#build-sparse-frequency-vectors}

The next step is to create a histogram of the visual features for each image. The histogram is a sparse vector, where each element represents the count of a visual feature in the image. The histogram is then normalized to create a feature vector. Given an input image, the feature vector is extracted and assigned a label based on the cluster model. That label is one of the \\(n\\) chosen words in the vocabulary, which is incremented in the histogram.


### Adjust Frequency Vectors {#adjust-frequency-vectors}

The feature vectors are then adjusted to account for the frequency of the visual features. This is done by applying a weighting scheme to the feature vectors. The most common weighting scheme is the Term Frequency-Inverse Document Frequency (TF-IDF) weighting. This weighting scheme is used to adjust the frequency of a word in a document based on the frequency of the word in the entire dataset. In terms of the Bag of Visual Words, the frequency of a visual feature in an image is adjusted based on the frequency of the visual feature in the entire dataset.


### Compare Vectors {#compare-vectors}

The last step is to compare the feature vectors in service of some downstream task like classification. Since every feature vector is a fixed length, they can be used as input to a machine learning model.
