+++
title = "Deep Learning"
authors = ["Alex Dillhoff"]
date = 2022-03-29T00:00:00-05:00
tags = ["deep learning"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Syllabi](#syllabi)
- [Questions](#questions)
- [Introduction](#introduction)
- [What makes a model deep?](#what-makes-a-model-deep)
- [Deep Networks](#deep-networks)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [A Typical Training Pipeline](#a-typical-training-pipeline)
- [Bias/Variance Tradeoff](#bias-variance-tradeoff)
- [Regularization](#regularization)
- [Optimization Algorithms](#optimization-algorithms)

</div>
<!--endtoc-->



## Syllabi {#syllabi}

-   <https://cs230.stanford.edu/syllabus/>


## Questions {#questions}

-   Does backprop give the same result when combining the weights and bias versus separating?


## Introduction {#introduction}

-   What is deep learning?
-   What problems is it used for?
-   Basic neural nets vs "deep" nets
-   Structuring a project
-   CNNs
-   Sequence models
-   Geometric Deep Learning
-   Representation of a model in code (vectorization)

Deep learning has been employed in many fields from visual recognition to medical diagnosis.
Its continued success is impossible to ignore.
As more and more companies pivot to utilize deep learning in some capacity, it is paramount that any machine learning practitioner or research be well versed in moden practices and frameworks.
This section will provide a broad overview of deep learning with examples using [PyTorch](https://pytorch.org).

If you are not yet familiar with [neural networks]({{< relref "neural_networks.md" >}}), follow the link to learn about their basics as they are the foundation of deep learning systems.

We will cover how to implement an array of deep learning models for different tasks.
Different layers and activation functions will be explored as well as the effect of regularization.
There will also be a focus on best practices for organizing a machine learning project.


## What makes a model deep? {#what-makes-a-model-deep}

We begin by comparing _shallow_ networks with _deep_ networks.
What defines a deep network? Is it as simple as crossing a threshold into \\(n\\) layers?
As evidenced by <&zeilerVisualizingUnderstandingConvolutional2013>, deeper networks allow for a more robust hierarchy of image features.

There is work by <&montufarNumberLinearRegions2014> which suggests that shallow networks require an exponential amount of nodes as compared to deeper networks.
Additionally, there are many individual results which suggest that deeper networks provide better task generalization.

As we will later see when studying Convolutional Neural Networks, the optimization of such deep networks produces features that maximize the performance of a task.
That is, our network is not only optimizing the performance of task, but it produces features from the data.
This is particularly useful for transfer learning, where large pre-trained models can be used as starting points for novel tasks.
The benefit being that a complete retraining of the model is not necessary.


## Deep Networks {#deep-networks}

Like [neural networks]({{< relref "neural_networks.md" >}}), deep networks are defined by the number of layers, nodes per layer, activation functions, and loss functions.
We now review the forward and backward pass, providing more insight into the structure and usage of deep networks along the way.

Consider a deep network with \\(L\\) layers. Layer \\(l\\) has \\(n\_{l-1}\\) input connections and \\(n\_l\\) output nodes and activation function \\(g^{(l)}\\).
The final output is evaluated with some ground truth using a loss function \\(\mathcal{L}\\).


### Forward Pass {#forward-pass}

\begin{align\*}
\mathbf{a}^{(l)} &= W^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}\\\\
\mathbf{z}^{(l)} &= g^{(l)}(\mathbf{a}^{(l)})\\\\
\end{align\*}

This is repeated from the input to the last layer.
For the first layer \\(l=1\\), the input \\(\mathbf{z}^{(0)} = \mathbf{x}\\).
In practice, the output \\(\mathbf{a}^{(l)}\\) is cached since it is required for the backward pass.
This prevents the values from needing to be computed twice.

It is also worth it to study the sizes of the matrices while performing a forward pass.
For a layer \\(l\\), \\(W^{(l)} \in \mathbb{R}^{n\_l \times n\_{l-1}}\\) and the input \\(\mathbf{z}^{(l-1)} \in \mathbb{R}^{n\_{l-1} \times 1}\\).
When training, it is common to perform batch gradient descent with batches of input of size \\(B\\).
Then, \\(\mathbf{z}^{(l-1)} \in \mathbb{R}^{n\_{l-1}\times B}\\) and \\(\mathbf{a}^{(l)}, \mathbf{b}^{(l)} \in \mathbb{R}^{n\_l \times B}\\).


### Backward Pass {#backward-pass}

During the backward pass, the gradient is propagated from the last layer to the first.
Each layer that contains trainable parameters must also compute the gradient of the network output with respect to the weights and biases.
This can be done in a modular way, as shown next.

Consider the last layer. The gradients with respect to the weights and biases are

\begin{align\*}
\frac{d\mathcal{L}}{dW^{(L)}} &= \frac{d\mathcal{L}}{d\mathbf{z}^{(L)}} \frac{d\mathbf{z}^{(L)}}{d\mathbf{a}^{(L)}} \frac{d\mathbf{a}^{(L)}}{dW^{(L)}}\\\\
\frac{d\mathcal{L}}{d\mathbf{b}^{(L)}} &= \frac{d\mathcal{L}}{d\mathbf{z}^{(L)}} \frac{d\mathbf{z}^{(L)}}{d\mathbf{a}^{(L)}} \frac{d\mathbf{a}^{(L)}}{d\mathbf{b}^{(L)}}.
\end{align\*}

To see how the gradient continues to be propagated backward, compute the same thing for layer \\(L-1\\)

\begin{align\*}
\frac{d\mathcal{L}}{dW^{(L-1)}} &= \frac{d\mathcal{L}}{d\mathbf{z}^{(L)}} \frac{d\mathbf{z}^{(L)}}{d\mathbf{a}^{(L)}} \frac{d\mathbf{a}^{(L)}}{d\mathbf{z}^{(L-1)}} \frac{d\mathbf{z}^{(L-1)}}{d\mathbf{a}^{(L-1)}} \frac{d\mathbf{a}^{(L-1)}}{dW^{(L-1)}}\\\\
\frac{d\mathcal{L}}{d\mathbf{b}^{(L-1)}} &= \frac{d\mathcal{L}}{d\mathbf{z}^{(L)}} \frac{d\mathbf{z}^{(L)}}{d\mathbf{a}^{(L)}} \frac{d\mathbf{a}^{(L)}}{d\mathbf{z}^{(L-1)}} \frac{d\mathbf{z}^{(L-1)}}{d\mathbf{a}^{(L-1)}} \frac{d\mathbf{a}^{(L-1)}}{d\mathbf{b}^{(L-1)}}.
\end{align\*}

As seen above, to continue propagating the gradient backward, each layer \\(l\\) must also compute

\\[
\frac{d\mathbf{a}^{(l)}}{d\mathbf{z}^{(l-1)}}.
\\]

To summarize, every layer with trainable parameters will compute

\begin{align\*}
\frac{d\mathcal{L}}{dW^{(l)}} = \frac{d\mathbf{a}^{(l+1)}}{d\mathbf{z}^{(l)}} \frac{d\mathbf{z}^{(l)}}{d\mathbf{a}^{(l)}} \frac{d\mathbf{a}^{(l)}}{dW^{(l)}}\\\\
\frac{d\mathcal{L}}{d\mathbf{b}^{(l)}} = \frac{d\mathbf{a}^{(l+1)}}{d\mathbf{z}^{(l)}} \frac{d\mathbf{z}^{(l)}}{d\mathbf{a}^{(l)}} \frac{d\mathbf{a}^{(l)}}{d\mathbf{b}^{(l)}}.
\end{align\*}

The term \\(\frac{d\mathbf{a}^{(l+1)}}{d\mathbf{z}^{(l)}}\\) is the gradient that is propagated from layer \\(l+1\\).


## Activation Functions {#activation-functions}


### Sigmoid {#sigmoid}

**Function**

\\[
\sigma(\mathbf{x}) = \frac{1}{1 + \exp(-\mathbf{x})}
\\]

**Derivative**

\\[
\sigma(\mathbf{x})(1 - \sigma(\mathbf{x}))
\\]

{{< figure src="/ox-hugo/2022-03-31_10-04-44_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Sigmoid non-linearity (Wikipedia)" >}}


## Loss Functions {#loss-functions}


## A Typical Training Pipeline {#a-typical-training-pipeline}

When training and evaluating models, especially on benchmark datasets, it is important to properly test their generalization performance.
This test is crucial when comparing the efficacy of your ideas versus baseline evaluations or competing methods.

To ensure that your model is evaluated in a fair way, it is common to set aside a set of test data that is only used during the final comparison.
This data is typically annotated so that some metric can be used.

It is true that the training data drives the parameter tuning during optimization.
This is most commonly done with gradient descent.
However, we will also change the hyperparamers such as learning rate, batch size, and data augmentation.
In this case, we want to evaluate the relative performance of each change.

If we use the test set to do this, then we are necessarily using the test set for training.
Our biases and intuitions about the model's performance would be implicitly influenced by that set.
To track our relative changes without using the test set, we can take a portion of the original training set and label it as our **validation set**.

The split between training, validation, and test data is relatively small.
Most modern datasets are large, with millions of samples.
Consider [ImageNet](https://www.image-net.org/), an image classification dataset with over 14 million samples.
Taking 10,000 samples to serve as a validation set is only \\(~.07\\%\\) of the dataset.

Most modern machine learning frameworks have an easy way to split the dataset.
We can do this in PyTorch using `torch.utils.data.random_split`.

```python
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
```


## Bias/Variance Tradeoff {#bias-variance-tradeoff}

Ideally, we want a model that generalizes well to unseen data.
Being able to evaluate how well our model performs is, in many ways, more important than the model itself.
A model that shows good generalization performance will have low bias and low variance.
Bonus points are awarded if the model remains simple.

During training, we can detect a model that is overfitting the data by also monitoring its performance on a separate validation set.
If the validation loss diverges from the training loss, the model is beginning to overfit.

{{< figure src="/ox-hugo/2022-03-31_22-38-54_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Higher complexity on the dataset leads to higher variance and lower bias." >}}


## Regularization {#regularization}


## Optimization Algorithms {#optimization-algorithms}
