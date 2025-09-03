+++
title = "Boosting"
authors = ["Alex Dillhoff"]
date = 2022-03-23T00:00:00-05:00
tags = ["machine learning"]
draft = false
lastmod = 2025-09-03
sections = "Machine Learning"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [AdaBoost](#adaboost)

</div>
<!--endtoc-->



## Introduction {#introduction}

Combining predictions from multiple sources is usually preferred to a single source.
For example, a medical diagnosis would carry much more weight if it was the result of a consensus of several experts.
This idea of prediction by consensus is a powerful way to improve classification and regression models.
In fact, good performance of a committee of models can be achieved even if each individual model is conceptually very simple.

**Boosting** is one such way of building a committee of models for classification or regression and is popularly implemented by an algorithm called **AdaBoost**.


## AdaBoost {#adaboost}

Given a dataset \\(\\{\mathbf{x}\_i\\}\\) and target variables \\(\\{\mathbf{y}\_i\\}\\), AdaBoost first initializes a set of weights corresponding to each data sample as \\(w\_i = \frac{1}{N}\\).
At each step of the algorithm, a simple classifier, called a **weak learner** is fit to the data.
The weights for each sample are adjusted based on the individual classifier's performance.
If the sample was misclassified, the relative weight for that sample is increased.
After all classifiers have been fit, they are combined to form an ensemble model.


### Exponential Loss {#exponential-loss}

Gradient boosting is applied to binary classification using an exponential loss. In its original derivation, a target of \\(y \in \\{-1, +1\\}\\) is used. As in [Logistic Regression]({{< relref "logistic_regression.md" >}}), the likelihood of a Bernoulli distribution is used. That original criterion is

\\[
\mathbb{E}[\exp(-yF(x))] = P(y=1 \vert x)\exp(-F(x)) + P(y=-1|x)\exp(F(x)).
\\]

The primary motivation for this loss function is that it serves as a differentiable upper bound to misclassification error (Friedman, Jerome and Hastie, Trevor and Tibshirani, Robert, 2000).

This expectation is minimized at

\\[
F(x) = \frac{1}{2} \log \frac{P(y=1 \vert x)}{P(y=-1 \vert x)}.
\\]

Therefore,

\\[
P(y=1 \vert x) = \frac{\exp(F(x))}{\exp(-F(x)) + \exp(F(x))}.
\\]


### The Algorithm {#the-algorithm}

1.  Initialize data weights \\({w\_i}\\) as \\(w\_i^{(1)} = \frac{1}{n}\\) for \\(i = 1, \dots, n\\).
2.  Fit each weak learner \\(j\\) to the training data by minimizing the misclassification cost:

    \\[
       \sum\_{i=1}^n w\_i^{(j)} \mathbb{1}(f\_j(\mathbf{x}\_i) \neq \mathbf{y}\_i)
       \\]

3.  Compute a weighted error rate

    \\[
       \epsilon\_j = \frac{\sum\_{i=1}^n w\_i^{(j)} \mathbb{1}(f\_j(\mathbf{x}\_i) \neq \mathbf{y}\_i)}{\sum\_{i=1}^n w\_i^{(j)}}
       \\]

4.  Use the weighted error rate to compute a weight for each classifier such that misclassified samples are given higher weight:

    \\[
       \alpha\_j = \ln \bigg\\{\frac{1 - \epsilon\_j}{\epsilon\_j}\bigg\\}.
       \\]

5.  Update the data weights for the next model in the sequence:

    \\[
       w\_i^{j+1} = w\_i^{j} \exp\\{\alpha\_j \mathbb{1}(f\_j(\mathbf{x}\_i \neq \mathbf{y}\_i)\\}.
       \\]

Once all weak learners are trained, the final model predictions are given by

\\[
Y\_M(\mathbf{x}) = \text{sign} \Bigg(\sum\_{j=1}^M \alpha\_j f\_j(\mathbf{x})\Bigg).
\\]


### Weak Learners {#weak-learners}

The weak learners can be any classification or regression model.
However, they are typically chosen to be very simple to account for training time.
For example, a complex deep learning model would be a poor choice for a weak learner.

One example of a weak learner is a simple linear model like a [Perceptron]({{< relref "perceptron.md" >}}) or decision stump.
A standard implementation of AdaBoost uses a decision tree with depth 1, as observed in [sklearn's implementation.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html?highlight=boost#sklearn.ensemble.AdaBoostClassifier)


### Example {#example}

Let's put this together and walk through the first few steps of training an AdaBoost model using a decision stump as the weak learner. We will use a very simple dataset to keep the values easy to compute by hand.

**Initial Data**

| x1 | x2 | y | weight |
|----|----|---|--------|
| 1  | 5  | 0 | 0.2    |
| 2  | 6  | 1 | 0.2    |
| 3  | 7  | 0 | 0.2    |
| 4  | 8  | 1 | 0.2    |
| 5  | 9  | 1 | 0.2    |

**Weak Learner 1**

The first learner is trained on the initial data and picks \\(x\_1 = 2.5\\) as the split threshold.
Input where \\(x\_1 \leq 2.5\\) is assigned to class 0 and all other samples are assigned class 1.
The data with this learner's predictions are shown below.

| x1 | x2 | y | weight | prediction |
|----|----|---|--------|------------|
| 1  | 5  | 0 | 0.2    | 0          |
| 2  | 6  | 1 | 0.2    | 0          |
| 3  | 7  | 0 | 0.2    | 1          |
| 4  | 8  | 1 | 0.2    | 1          |
| 5  | 9  | 1 | 0.2    | 1          |

**Error and weight**

The error is simple enough to compute as all samples are currently weighted equally. Since two of the samples were misclassified, the error is the sum of their weights.

**Total error**

\\(e\_1 = 0.2 + 0.2 = 0.4\\).

The weight of the classifier can then be computed.

**Classifier weight**

\\(\alpha\_1 = \frac{1}{2} \ln \big(\frac{1 - e\_1}{e\_1}\big) = 0.2027\\).

The weights of our data can now be updated using this value of \\(\alpha\_1\\).
The weight of each example is updated by multiplying each correctly classifed sample by \\(\exp\\{-\alpha\_1\\}\\) and each incorrectly classified sample by \\(\exp\\{\alpha\\}\\):

\\[
w\_i^{j+1} = w\_i^{j} \exp\\{\alpha\_j \mathbb{1}(f\_j(\mathbf{x}\_i \neq \mathbf{y}\_i)\\}.
\\]

**NOTE:** You will notice that the equation above is different from the actual update rule that was applied to the weights in this example. In the original publication, the weights are renormalized at the end of the loop (Friedman, Jerome and Hastie, Trevor and Tibshirani, Robert, 2000). In this example, the normalization is combined with the update. In either case, the updated weights are shown below.

| x1 | x2 | y | weight |
|----|----|---|--------|
| 1  | 5  | 0 | 0.167  |
| 2  | 6  | 1 | 0.250  |
| 3  | 7  | 0 | 0.250  |
| 4  | 8  | 1 | 0.167  |
| 5  | 9  | 1 | 0.167  |

**Weak Learner 2**

The algorithm now moves to the next weak learner, which classifies the data given a threshold of \\(x\_1 = 3.5\\). Its predictions are shown below.

| x1 | x2 | y | weight | prediction |
|----|----|---|--------|------------|
| 1  | 5  | 0 | 0.167  | 0          |
| 2  | 6  | 1 | 0.250  | 0          |
| 3  | 7  | 0 | 0.250  | 0          |
| 4  | 8  | 1 | 0.167  | 1          |
| 5  | 9  | 1 | 0.167  | 1          |

Only a single sample is misclassified, and the error is computed as before.

**Total error**

\\(e\_2 = 0.250\\)

**Classifier weight**

\\(\alpha\_2 = \frac{1}{2} \ln \big(\frac{1 - e\_2}{e\_2}\big) = 0.5493\\)

The weights are updated for each sample, yielding the following data:

| x1 | x2 | y | weight |
|----|----|---|--------|
| 1  | 5  | 0 | 0.111  |
| 2  | 6  | 1 | 0.500  |
| 3  | 7  | 0 | 0.167  |
| 4  | 8  | 1 | 0.111  |
| 5  | 9  | 1 | 0.111  |

The second sample has been misclassified twice at this point, leading to a relatively high weight. This will hopefully be addressed by the third learner.

**Weak Learner 3**

The final weak learner splits the data on \\(x\_2 = 6.5\\), yielding the following output for each sample.

| x1 | x2 | y | weight | prediction |
|----|----|---|--------|------------|
| 1  | 5  | 0 | 0.111  | 0          |
| 2  | 6  | 1 | 0.500  | 0          |
| 3  | 7  | 0 | 0.167  | 1          |
| 4  | 8  | 1 | 0.111  | 1          |
| 5  | 9  | 1 | 0.111  | 1          |

Unfortunately, sample 2 is too tricky for any of our weak learners. The total error is shown below. Since this is a binary classification problem, the error suggests that our weak learner performs worse than random guessing.

**Total error**

\\(e\_3 = 0.667\\)

**Classifier weight**

\\(\alpha\_3 = \frac{1}{2} \ln \big(\frac{1 - e\_3}{e\_3}\big) = -0.3473\\)

The negative value of the classifier weight suggests that its predictions will be reversed when evaluated. The updated weights of each data sample are given below.

| x1 | x2 | y | weight |
|----|----|---|--------|
| 1  | 5  | 0 | 0.167  |
| 2  | 6  | 1 | 0.375  |
| 3  | 7  | 0 | 0.125  |
| 4  | 8  | 1 | 0.167  |
| 5  | 9  | 1 | 0.167  |

**Final Classifier**

The final classifier is a weighted vote of the three weak learners, with the weights being the classifier weights we calculated (0.2027, 0.5493, and -0.3473). The negative weight means that the third learner's predictions are reversed.

## References

Friedman, Jerome and Hastie, Trevor and Tibshirani, Robert (2000). _Additive Logistic Regression: A Statistical View of Boosting ({{With}} Discussion and a Rejoinder by the Authors)_, Institute of Mathematical Statistics.
