+++
title = "Automatic Differentiation"
authors = ["Alex Dillhoff"]
date = 2024-09-23T18:00:00-05:00
tags = ["machine learning", "optimization"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Types of Differentiation](#types-of-differentiation)
- [Forward Mode AD](#forward-mode-ad)
- [Reverse Mode AD](#reverse-mode-ad)
- [Basic Implementation in Python](#basic-implementation-in-python)
- [Matrix Implementation](#matrix-implementation)
- [Comparison with PyTorch](#comparison-with-pytorch)

</div>
<!--endtoc-->



## Introduction {#introduction}

These notes largely follow the survey presented by (<a href="#citeproc_bib_item_1">Baydin et al. 2018</a>). I have added a few examples to clarify the matrix algebra as well as a lead in to a practical implemenation.

Automatic differentiation is a method for computing the derivatives of functions in a modular way using the chain rule of calculus. It is used in many deep learning frameworks such as PyTorch and Tensorflow. Consider a complex series of functions that together work to yield some useful input, such as that of a deep learning model. Traditionally, the parameters of such a model would be optimized through gradient descent. This requires that the derivatives with respect to the parameters are implemented for every function used in the model.

Without autograd, one would have to write implementations for each function derivative required for the model. This is not only cumbersome, but can be error-prone and hard to debug. Automatic differentiation instead breaks down any forward computation into a series of elementary operations whose derivatives are well known.


## Types of Differentiation {#types-of-differentiation}


### Numerical Differentiation {#numerical-differentiation}


### Symbolic Differentiation {#symbolic-differentiation}


## Forward Mode AD {#forward-mode-ad}


### Computing the Jacobian {#computing-the-jacobian}

The example above computed the derivative of a function \\(f : \mathbb{R}^2 \to \mathbb{R}\\). To compute the derivative of the function with respect to all input variables, we would need to run forward mode AD for each input variable. In the context of a neural network, this would mean running forward mode AD for each input feature. This is not efficient, as the number of input features can be very large. What about hidden layers? They typically have multiple outputs as well. Luckily, the number of outputs is not a bottleneck as this approach can compute \\(\dot{y}\_j\\) for all \\(j\\) in a single pass by setting \\(\dot{\mathbf{x}} = \mathbf{e}\_i\\) where \\(\mathbf{e}\_i\\) is the $i$th standard basis vector.

If we combine these columns representing the derivatives of the output with respect to the input, we get the **Jacobian matrix**. This matrix is the derivative of the function with respect to all input variables. This is a very useful matrix in optimization problems, as it tells us how the output of the function changes with respect to the input.


## Reverse Mode AD {#reverse-mode-ad}


## Basic Implementation in Python {#basic-implementation-in-python}


## Matrix Implementation {#matrix-implementation}


## Comparison with PyTorch {#comparison-with-pytorch}

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Baydin, Atilim Gunes, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. 2018. “Automatic Differentiation in Machine Learning: A Survey.” arXiv. <a href="https://doi.org/10.48550/arXiv.1502.05767">https://doi.org/10.48550/arXiv.1502.05767</a>.</div>
</div>
