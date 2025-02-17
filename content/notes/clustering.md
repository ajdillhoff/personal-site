+++
title = "Clustering"
authors = ["Alex Dillhoff"]
date = 2025-02-16T00:00:00-06:00
tags = ["computer vision"]
draft = false
lastmod = 2025-02-16
sections = "Computer Vision"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [K-Means Clustering](#k-means-clustering)
- [K-Medoids Clustering](#k-medoids-clustering)
- [Mixtures of Gaussians](#mixtures-of-gaussians)

</div>
<!--endtoc-->

In machine learning, the three most common forms of learning are

1.  supervised learning,
2.  unsupervised learning,
3.  and reinforcement learning.

Supervised learning involves training a model on a set of inputs with matching ground truth signals. Regression and classification models are most commonly trained via supervised learning, but it is possible to train without labeled samples. [Reinforcement Learning]({{< relref "reinforcement_learning.md" >}}) optimizes an agent's behavior based on a reward signal. The goal of ****unsupervised learning**** is to extract some structure from the data itself. Models that are trained via unsupervised learning can also be used for regression and classification. If you fully understand the data generating distribution, surely you could answer any other question about that data.

**Clustering** is a common approach to unsupervised learning. Instead of predicting a target value, clustering algorithms group similar data points together based on a similarity metric. The choice of metric depends on the data and the problem. For example, clustering for images may consider spatial proximity as well as pixel values.


## K-Means Clustering {#k-means-clustering}

Imagine you are given a database of houses with features like square footage, number of bedrooms, number of bathrooms, and, of course, the price itself. Knowing that the price of the house varies greatly depending on these factors and others such as location, you decide to group the data depending on the cost of the house. This way, your customers can browse houses in their price range.

To keep things simple, you choose to group the houses into three different tiers: low cost, medium cost, and high cost. Given the larger number of samples, this would be a perfect application of K-Means. First, the cluster centers are initialized randomly. The algorithm assigns each data point to a cluster center based on the Euclidean distance. The centers are then updated based on the samples that were assigned to them.

1.  For each sample, calculate the Euclidean distance to each cluster center.
    \\[
       z\_n^\* = \arg\min\_{z\_n} \sum\_{k=1}^K ||x\_n - \mu\_k||^2
       \\]
2.  Update the cluster centers.
    \\[
        \mu\_k = \frac{1}{N\_k} \sum\_{n:z\_n=k} x\_n
        \\]
3.  Repeat steps 1 and 2 until convergence.

That's it! K-Means is a simple algorithm that can be implemented in a few lines of code. It runs in \\(O(NKT)\\) time, where \\(T\\) is the number of iterations. In practice, you can affect the number of iterations by monitoring the **distortion** from one iteration to the next.

\\[
J = \sum\_{n=1}^N ||x\_n - \mu\_{z\_n}||^2
\\]

This measure is evaluating points against the cluster they are assigned to. You should be able to convince yourself that, as the number of clusters increases, the distortion decreases. When decided when to converge, you can monitor the change in distortion from one iteration to the next. If the change is less than some threshold, the loop will stop.


### Inference {#inference}

Once the algorithm converges, you can assign new samples to the cluster centers. This is done by calculating the Euclidean distance to each cluster center and assigning the sample to the closest one. The cluster centers themselves can be used as a summary of the data. In our case, each cluster center represents the average house price in that range.

Inference is much cheaper than training, you only need to consider the distance to each cluster center.


## K-Medoids Clustering {#k-medoids-clustering}

As long as our samples can be compared via Euclidean distance, K-Means is a great choice. However, if the data is not continuous, or if each sample has a varying number of features, K-Medoids is the way to go. In this variant, the cluster center is chosen from the sample whose average dissimilarity to the other samples is the smallest.

For each sample, find the closest cluster center.

\\[
z\_n = \arg\min\_{z\_n} d(x\_n, m\_k)\quad \forall n
\\]

For each cluster, find the sample whose total distance to all other samples in the cluster is the smallest.

\\[
m\_k = \arg\min\_{n:z\_n = k} \sum\_{n':z\_{n'}=k} d(x\_n, x\_{n'})\quad \forall k
\\]

This would be a great choice for clustering text data, where the distance between two samples is not as straightforward as Euclidean distance.


## Mixtures of Gaussians {#mixtures-of-gaussians}

Let's go back to our housing example. Suppose we have a K-Means model that groups houses into three clusters. If we come across a new sample, we can assign it to the closest cluster center. However, this sample may not be a perfect fit for any of the clusters. The only way we can measure the fitness of a new sample is by calculating the Euclidean distance to each cluster center. How do we decide if the cluster belongs or not? A naive solution is to set a predetermined threshold. If the distance is less than the threshold, the sample belongs to the cluster. If we want something that is more robust, we can use a probabilistic model.

A Gaussian mixture model is defined as a weighted sum of Gaussian distributions.

\\[
p(x) = \sum\_{k=1}^K \pi\_k \mathcal{N}(x|\mu\_k, \Sigma\_k)
\\]

If we know the parameters of each Gaussian, we can answer the question of which Gaussian best explains a given data sample:

\\[
p(k|x) = \frac{\pi\_k \mathcal{N}(x|\mu\_k, \Sigma\_k)}{\sum\_{k'=1}^K \pi\_{k'} \mathcal{N}(x|\mu\_{k'}, \Sigma\_{k'})}
\\]

To make an assignment, compute the probability of each cluster given the sample and assign it to the cluster with the highest probability. This can be done in a more computationally efficient manner by only considering the log of the numerator.

\\[
z\_n = \arg\max\_{k} \log \pi\_k + \log \mathcal{N}(x\_n|\mu\_k, \Sigma\_k)
\\]
