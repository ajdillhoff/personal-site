+++
title = "Linear Discriminant Analysis"
authors = ["Alex Dillhoff"]
date = 2022-01-22T00:00:00-06:00
draft = false
lastmod = 2023-08-31
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Gaussian Class Conditional Densities](#gaussian-class-conditional-densities)
- [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
- [Quadratic Descriminant Analysis](#quadratic-descriminant-analysis)
- [Example](#example)

</div>
<!--endtoc-->



## Introduction {#introduction}

This section covers classification from a probabilistic perspective.
The [discriminative approach]({{< relref "discriminant_functions.md" >}}) involves a parameterized function which assigns each input vector \\(\mathbf{x}\\) to a specific class.
We will see that modeling the conditional probability distribution \\(p(C\_k|\mathbf{x})\\) grants us additional benefits while still fulfilling our original classification task.

Let's begin with a 2 class problem. To classify this with a generative model, we use the class-conditional densities \\(p(\mathbf{x}|C\_i)\\) and class priors \\(p(C\_i)\\).
The posterior probability for \\(C\_1\\) can be written in the form of a sigmoid function:

\begin{align\*}
p(C\_1|\mathbf{x}) &= \frac{p(\mathbf{x}|C\_1)p(C\_1)}{p(\mathbf{x}|C\_1)p(C\_1) + p(\mathbf{x}|C\_2)p(C\_2)}
\end{align\*}

Then multiply the numerator and denominator by

\begin{equation\*}
\frac{(p(\mathbf{x}|C\_1))^{-1}}{(p(\mathbf{x}|C\_1))^{-1}},
\end{equation\*}

which yields

\begin{equation\*}
\frac{1}{1 + \frac{p(\mathbf{x}|C\_2)p(C\_2)}{p(\mathbf{x}|C\_1)p(C\_1)}}.
\end{equation\*}

Noting that \\(a = \exp(\ln(a))\\), we can rewrite further

\begin{equation\*}
\frac{1}{1 + \exp(-a)},
\end{equation\*}

where \\(a = \ln \frac{p(\mathbf{x}|C\_1)p(C\_1)}{p(\mathbf{x}|C\_2)p(C\_2)}\\).

Writing this distribution in the form of the sigmoid function is convenient as it is a natural choice for many other classification models. It also has a very simple derivative which is convenient for models optimized using gradient descent.

Given certain choices for the class conditional densities, the posterior probabilty distribution will be a linear function of the input features:

\begin{equation\*}
\ln p(C\_k|\mathbf{x};\theta) = \mathbf{w}^T \mathbf{x} + c,
\end{equation\*}

where \\(\mathbf{w}\\) is a parameter vector based on the parameters of the chosen probability distribution, and \\(c\\) is a constant term that is not dependent on the parameters. As we will see, the resulting model will take an equivalent form to the discriminative approach.


## Gaussian Class Conditional Densities {#gaussian-class-conditional-densities}

Let's assume that our class conditional densities \\(p(\mathbf{x}|C\_k)\\) are Gaussian. We will additionally assume that the covariance matrices between classes are shared. This will result in linear decision boundaries. Since the conditional densities are chosen to be Gaussian, the posterior is given by

\begin{equation\*}
p(C\_k|\mathbf{x};\theta) \propto \pi\_k\mathcal{N}(\mathbf{x}|\mathbf{\mu}\_c,\Sigma),
\end{equation\*}

where \\(\pi\_k\\) is the prior probability of class \\(k\\). We choose to ignore the normalizing constant since it is not dependent on the class.

The class conditional density function for class \\(k\\) is given by

\begin{equation\*}
p(\mathbf{x}|C\_k;\theta) = \frac{1}{2\pi^{D/2}}\frac{1}{|\Sigma|^{1/2}}\exp\Big(-\frac{1}{2}(\mathbf{x} - \mathbf{\mu}\_k)^T \Sigma^{-1} (\mathbf{x} - \mathbf{\mu}\_k)\Big).
\end{equation\*}

Now that we have a concrete function to work with, let's go back to the simple case of two classes and define \\(a = \ln \frac{p(\mathbf{x}|C\_1)p(C\_1)}{p(\mathbf{x}|C\_2)p(C\_2)}\\). First, we rewrite \\(a\\):

\begin{equation\*}
a = \ln p(\mathbf{x}|C\_1) - \ln p(\mathbf{x}|C\_2) + \ln \frac{p(C\_1)}{p(C\_2)}.
\end{equation\*}

The log of the class conditional density for a Gaussian is

\begin{equation\*}
\ln p(\mathbf{x}|C\_k;\mathbf{\mu}\_k,\Sigma) =
-\frac{D}{2}\ln(2\pi) - \frac{1}{2}\ln|\Sigma|-\frac{1}{2}(\mathbf{x}-\mathbf{\mu}\_k)^T \Sigma^{-1} (\mathbf{x}-\mathbf{\mu}\_k).
\end{equation\*}

To simplify the above result, we will group the terms that are not dependent on the class parameters since they are consant:

\begin{equation\*}
\ln p(\mathbf{x}|C\_k;\mathbf{\mu}\_k,\Sigma) =
-\frac{1}{2}(\mathbf{x}-\mathbf{\mu}\_k)^T \Sigma^{-1} (\mathbf{x}-\mathbf{\mu}\_k) + c.
\end{equation\*}

Observing that this quantity takes on a quadratic form, we can rewrite the above as

\begin{equation\*}
\ln p(\mathbf{x}|C\_k;\mathbf{\mu}\_k,\Sigma) =
-\frac{1}{2}\mathbf{\mu}\_k\Sigma^{-1}\mathbf{\mu}\_k + \mathbf{x}^T \Sigma^{-1} \mathbf{\mu}\_k
-\frac{1}{2}\mathbf{x}^T \Sigma^{-1}\mathbf{x} + c.
\end{equation\*}

Using this, we complete the definition of \\(a\\):

\begin{align\*}
a &= \ln p(\mathbf{x}|C\_1) - \ln p(\mathbf{x}|C\_2) + \ln \frac{p(C\_1)}{p(C\_2)}\\\\
&= -\frac{1}{2}\mathbf{\mu}\_1\Sigma^{-1}\mathbf{\mu}\_1 + \mathbf{x}^T \Sigma^{-1} \mathbf{\mu}\_1 + \frac{1}{2}\mathbf{\mu}\_2\Sigma^{-1}\mathbf{\mu}\_2 - \mathbf{x}^T \Sigma^{-1} \mathbf{\mu}\_2 + \ln \frac{p(C\_1)}{p(C\_2)}\\\\
&= \mathbf{x}^T(\Sigma^{-1}(\mathbf{\mu}\_1 - \mathbf{\mu}\_2)) - \frac{1}{2}\mathbf{\mu}\_1\Sigma^{-1}\mathbf{\mu}\_1 + \frac{1}{2}\mathbf{\mu}\_2\Sigma^{-1}\mathbf{\mu}\_2 + \ln \frac{p(C\_1)}{p(C\_2)}\\\\
&= (\Sigma^{-1}(\mathbf{\mu}\_1 - \mathbf{\mu}\_2))^T \mathbf{x} - \frac{1}{2}\mathbf{\mu}\_1\Sigma^{-1}\mathbf{\mu}\_1 + \frac{1}{2}\mathbf{\mu}\_2\Sigma^{-1}\mathbf{\mu}\_2 + \ln \frac{p(C\_1)}{p(C\_2)}.
\end{align\*}

Finally, we define

\begin{equation\*}
\mathbf{w} = \Sigma^{-1}(\mathbf{\mu}\_1 - \mathbf{\mu}\_2)
\end{equation\*}

and

\begin{equation\*}
 w\_0 = - \frac{1}{2}\mathbf{\mu}\_1\Sigma^{-1}\mathbf{\mu}\_1 - \frac{1}{2}\mathbf{\mu}\_2\Sigma^{-1}\mathbf{\mu}\_2 + \ln \frac{p(C\_1)}{p(C\_2)}.
\end{equation\*}

Thus, our posterior takes on the form

\begin{equation\*}
p(C\_1|\mathbf{x};\theta) = \sigma(\mathbf{w}^T \mathbf{x} + w\_0).
\end{equation\*}


### Multiple Classes {#multiple-classes}

What if we have more than 2 classes?
Recall that a **generative classifier** is modeled as

\\[
p(C\_k|\mathbf{x};\mathbf{\theta}) = \frac{p(C\_k|\mathbf{\theta})p(\mathbf{x}|C\_k, \mathbf{\theta})}{\sum\_{k'}p(C\_{k'}|\mathbf{\theta})p(\mathbf{x}|C\_{k'}, \mathbf{\theta})}.
\\]

As stated above, \\(\mathbf{\pi}\_k = p(C\_k|\mathbf{\theta})\\) and \\(p(\mathbf{x}|C\_k,\mathbf{\theta}) = \mathcal{N}(\mathbf{x}|\mathbf{\mu}\_c,\Sigma)\\).

For LDA, the covariance matrices are shared across all classes.
This permits a simplification of the class posterior distribution \\(p(C\_k|\mathbf{x};\mathbf{\theta})\\):

\begin{align\*}
p(C\_k|\mathbf{x};\mathbf{\theta}) &\propto \mathbf{\pi}\_k \exp\big(\mathbf{\mu}\_k^T \mathbf{\Sigma}^{-1}\mathbf{x} - \frac{1}{2}\mathbf{x}^T\mathbf{\Sigma}^{-1}\mathbf{x} - \frac{1}{2}\mathbf{\mu}\_k\mathbf{\Sigma}^{-1}\mathbf{\mu}\_k\big)\\\\
&= \exp\big(\mathbf{\mu}\_k^T \mathbf{\Sigma}^{-1}\mathbf{x}  - \frac{1}{2}\mathbf{\mu}\_k\mathbf{\Sigma}^{-1}\mathbf{\mu}\_k + \log \mathbf{\pi}\_k \big) \exp\big(- \frac{1}{2}\mathbf{x}^T\mathbf{\Sigma}^{-1}\mathbf{x}\big).
\end{align\*}

The term \\(\exp\big(- \frac{1}{2}\mathbf{x}^T\mathbf{\Sigma}^{-1}\mathbf{x}\big)\\) is placed aside since it is not dependent on the class \\(k\\).
When divided by the sum per the definition of \\(p(C\_k|\mathbf{x};\mathbf{\theta})\\), it will equal to 1.

Under this formulation, we let

\begin{align\*}
\mathbf{w}\_k &= \mathbf{\Sigma}^{-1}\mathbf{\mu}\_k\\\\
\mathbf{b}\_k &= -\frac{1}{2}\mathbf{\mu}\_k^T \mathbf{\Sigma}^{-1}\mathbf{\mu}\_k + \log \mathbf{\pi}\_k.
\end{align\*}

This lets us express \\(p(C\_k|\mathbf{x};\mathbf{\theta})\\) as the **softmax** function:

\\(p(C\_k|\mathbf{x};\mathbf{\theta}) = \frac{\exp(\mathbf{w}\_k^T\mathbf{x}+\mathbf{b}\_k)}{\sum\_{k'}\exp(\mathbf{w}\_{k'}^T\mathbf{x}+\mathbf{b}\_{k'})}\\).


## Maximum Likelihood Estimation {#maximum-likelihood-estimation}

Given our formulation in the previous section, we can estimate the parameters of the model via **maximum likelihood estimation**. Assuming \\(K\\) classes with Gaussian class conditional densities, the likelihood function is

\begin{equation\*}
p(\mathbf{X}|\mathbf{\theta}) = \prod\_{i=1}^n \mathcal{M}(y\_i|\mathbf{\pi})\prod\_{k=1}^K \mathcal{N}(\mathbf{x}\_i|\mathbf{\mu}\_k, \mathbf{\Sigma}\_k)^{\mathbb{1}(y\_i=k)}.
\end{equation\*}

Taking the log of this function yields

\begin{equation\*}
\ln p(\mathbf{X}|\mathbf{\theta}) = \Big[\sum\_{i=1}^n \sum\_{k=1}^K \mathbb{1}(y\_i=k)\ln \pi\_k\Big] + \sum\_{k=1}^K\Big[\sum\_{i:y\_i=c} \ln \mathcal{N}(\mathbf{x}\_n|\mathbf{\mu}\_k, \mathbf{\Sigma}\_k)\Big].
\end{equation\*}

Given that this is a sum of two different components, we can optimize the multinomial parameter \\(\mathbf{\pi}\\) and the class Gaussian parameters \\((\mathbf{\mu}\_k, \mathbf{\Sigma}\_k)\\) separately.


### Class Prior {#class-prior}

For multinomial distributions, the class prior parameter estimation \\(\hat{\pi}\_k\\) is easily calculated by counting the number of samples belonging to class \\(k\\) and dividing it by the total number of samples.

\\[
\hat{\pi}\_k = \frac{n\_k}{n}
\\]


### Class Gaussians {#class-gaussians}

The Gaussian parameters can be calculated as discussed during the probability review. The parameter estimates are

\begin{align\*}
\hat{\mathbf{u}}\_k &= \frac{1}{n\_k}\sum\_{i:y\_i=k}\mathbf{x}\_i\\\\
\hat{\Sigma}\_k &= \frac{1}{n\_k}\sum\_{i:y\_i=k}(\mathbf{x}\_i - \hat{\mathbf{\mu}}\_k)(\mathbf{x}\_i - \hat{\mathbf{\mu}}\_k)^T
\end{align\*}


### The Decision Boundary {#the-decision-boundary}

The decision boundary between two classes can be visualized at the point when \\(p(C\_k|\mathbf{x};\theta) = 0.5\\).


## Quadratic Descriminant Analysis {#quadratic-descriminant-analysis}

Linear Discriminant Analysis is a special case of Quadratic Discriminant Analysis (QDA) where the covariance matrices are shared across all classes. Assuming each class conditional density is Gaussian, the posterior probability is given by

\begin{equation\*}
p(C\_k|\mathbf{x};\theta) \propto \pi\_k\mathcal{N}(\mathbf{x}|\mathbf{\mu}\_k,\Sigma\_k).
\end{equation\*}

Taking the log of this function yields

\begin{equation\*}
\ln p(C\_k|\mathbf{x};\theta) = \ln \pi\_k - \frac{1}{2}\ln |\Sigma\_k| - \frac{1}{2}(\mathbf{x} - \mathbf{\mu}\_k)^T \Sigma\_k^{-1}(\mathbf{x} - \mathbf{\mu}\_k) + c.
\end{equation\*}

With LDA, the term \\(\frac{1}{2}\ln |\Sigma\_k|\\) is constant across all classes, so we treat it as another constant. Since QDA considers a different covariance matrix for each class, we must keep this term in the equation.


### Quadratic Decision Boundary {#quadratic-decision-boundary}

In the more general case of QDA, the decision boundary is quadratic, leading to a quadratic discriminant function. As shown above, the posterior probability function for LDA is linear in \\(\mathbf{x}\\), which leads to a linear discriminant function.


## Example {#example}

See [here](https://github.com/ajdillhoff/CSE6363/blob/main/logistic_regression/lda.ipynb) for an example using scikit-learn.
