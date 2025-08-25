+++
title = "Naive Bayes"
authors = ["Alex Dillhoff"]
date = 2022-01-22T00:00:00-06:00
tags = ["machine learning"]
draft = false
lastmod = 2025-08-25
sections = "Machine Learning"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Definition](#definition)
- [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
- [Making a Decision](#making-a-decision)
- [Relation to Multinomial Logistic Regression](#relation-to-multinomial-logistic-regression)
- [MNIST Example](#mnist-example)
- [Gaussian Formulation](#gaussian-formulation)

</div>
<!--endtoc-->

Slides for these notes can be found [here.](/teaching/cse6363/lectures/naive_bayes.pdf)


## Introduction {#introduction}

To motivate naive Bayes classifiers, let's look at slightly more complex data. The MNIST dataset was one of the standard benchmarks for computer vision classification algorithms for a long time. It remains useful for educational purposes. The dataset consists of 60,000 training images and 10,000 testing images of size \\(28 \times 28\\). These images depict handwritten digits. For the purposes of this section, we will work with binary version of the images. This implies that each data sample has 784 binary features.

We will use the naive Bayes classifier to make an image classification model which predicts the class of digit given a new image. Each image will be represented by a vector \\(\mathbf{x} \in \mathbb{R}^{784}\\). Modeling \\(p(\mathbf{x}|C\_k)\\) with a multinomial distribution would require \\(10 \times 2^{784} - 10\\) parameters since there are 10 classes and 784 features.

{{< figure src="/ox-hugo/2022-02-01_18-47-49_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Samples of the MNIST training dataset." >}}

With the naive assumption that the features are independent conditioned on the class, the number model parameters becomes \\(10 \times 784\\).


## Definition {#definition}

A naive Bayes classifier makes the assumption that the features of the data are independent. That is,
\\[
p(\mathbf{x}|C\_k, \mathbf{\theta}) = \prod\_{d=1}^D p(x\_i|C\_k, \theta\_{dk}),
\\]
where \\(\mathbf{\theta}\_{dk}\\) are the parameters for the class conditional density for class \\(k\\) and feature \\(d\\). Using the MNIST dataset, \\(\mathbf{\theta}\_{dk} \in \mathbb{R}^{784}\\). The posterior distribution is then

\begin{equation\*}
p(C\_k|\mathbf{x},\mathbf{\theta}) = \frac{p(C\_k|\mathbf{\pi})\prod\_{i=1}^Dp(x\_i|C\_k, \mathbf{\theta}\_{dk})}{\sum\_{k'}p(C\_{k'}|\mathbf{\pi})\prod\_{i=1}^Dp(x\_i|C\_{k'},\mathbf{\theta}\_{dk'})}.
\end{equation\*}

If we convert the input images to binary, the class conditional density \\(p(\mathbf{x}|C\_k, \mathbf{\theta})\\) takes on the Bernoulli pdf. That is,

\begin{equation\*}
p(\mathbf{x}|C\_k, \mathbf{\theta}) = \prod\_{i=1}^D\text{Ber}(x\_i|\mathbf{\theta}\_{dk}).
\end{equation\*}

The parameter \\(\theta\_{dk}\\) is the probability that the feature \\(x\_i=1\\) given class \\(C\_k\\).


## Maximum Likelihood Estimation {#maximum-likelihood-estimation}

Fitting a naive Bayes classifier is relatively simple using MLE. The likelihood is given by

\begin{equation\*}
p(\mathbf{X}, \mathbf{y}|\mathbf{\theta}) = \prod\_{n=1}^N \mathcal{M}(y\_n|\mathbf{\pi})\prod\_{d=1}^D\prod\_{k=1}^{K}p(x\_{nd}|\mathbf{\theta}\_{dk})^{\mathbb{1}(y\_n=k)}.
\end{equation\*}

To derive the estimators, we first take the log of the likelihood:

\begin{equation\*}
\ln p(\mathbf{X}, \mathbf{y}|\mathbf{\theta}) = \Bigg[\sum\_{n=1}^N\sum\_{k=1}^K \mathbb{1}(y\_n = k)\ln \pi\_k\Bigg] + \sum\_{k=1}^K\sum\_{d=1}^D\Bigg[\sum\_{n:y\_n=k}\ln p(x\_{nd}|\theta\_{dk})\Bigg].
\end{equation\*}

Thus, we have a term for the the multinomial and terms for the class-feature parameters. As with previous models that use a multinomial form, the parameter estimate for the first term is computed as

\begin{equation\*}
\hat{\pi}\_k = \frac{N\_k}{N}.
\end{equation\*}

The features used in our data are binary, so the parameter estimate for each \\(\hat{\theta}\_{dk}\\) follows the Bernoulli distribution:

\begin{equation\*}
\hat{\theta}\_{dk} = \frac{N\_{dk}}{N\_{k}}.
\end{equation\*}

That is, the number of times that feature \\(d\\) is in an example of class \\(k\\) divided by the total number of samples for class \\(k\\).


## Making a Decision {#making-a-decision}

Given parameters \\(\mathbf{\theta}\\), how can we classify a given data sample?

\begin{equation\*}
\text{arg}\max\_{k}p(y=k)\prod\_{i}p(x\_i|y=k)
\end{equation\*}


## Relation to Multinomial Logistic Regression {#relation-to-multinomial-logistic-regression}

Consider some data with discrete features having one of \\(K\\) states, then \\(x\_{dk} = \mathbb{1}(x\_d=k)\\). The class conditional density, in this case, follows a multinomial distribution:

\\[
p(\mathbf{x} \vert y = c, \mathbf{\theta}) = \prod\_{d=1}^D \prod\_{k=1}^K \theta\_{dck}^{x\_{dk}}.
\\]

We can see a connection between naive Bayes and logistic regression when we evaluate the posterior over classes:

\begin{align\*}
p(y=c|\mathbf{x}, \mathbf{\theta}) &= \frac{p(y=c)p(\mathbf{x}|y=c, \mathbf{\theta})}{p(\mathbf{x})}\\\\
&= \frac{\pi\_c \prod\_{d} \prod\_{k} \theta\_{dck}^{x\_{dk}}}{\sum\_{c'}\pi\_{c'}\prod\_{d}\prod\_{k}\theta\_{dc'k}^{x\_{dk}}} \\\\
&= \frac{\exp[\log \pi\_c + \sum\_d \sum\_k x\_{dk}\log \theta\_{dck}]}{\sum\_{c'} \exp[\log \pi\_{c'} + \sum\_d \sum\_k x\_{dk} \log \theta\_{dc'k}]}.
\end{align\*}

This has the same form as the softmax function:

\\[
p(y=c|\mathbf{x}, \mathbf{\theta}) = \frac{e^{\beta^{T}\_c \mathbf{x} + \gamma\_c}}{\sum\_{c'=1}^C e^{\beta^{T}\_{c'}\mathbf{x} + \gamma\_{c'}}}
\\]


## MNIST Example {#mnist-example}

With the model definition and parameter estimates defined, we can fit and evaluate the model. Using `scikit-learn`, we fit a Bernoulli naive Bayes classifier on the MNIST training set: [Naive Bayes](https://github.com/ajdillhoff/CSE6363/blob/main/logistic_regression/naive_bayes_mnist.ipynb).


## Gaussian Formulation {#gaussian-formulation}

If our features are continuous, we would model them with univariate Gaussians.
