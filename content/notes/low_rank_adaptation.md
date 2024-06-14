+++
title = "Low Rank Adaptation"
authors = ["Alex Dillhoff"]
date = 2024-06-08T13:03:00-05:00
tags = ["llms", "deep learning"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Key Concepts](#key-concepts)

</div>
<!--endtoc-->



## Key Concepts {#key-concepts}


### Traditional Fine-Tuning {#traditional-fine-tuning}

Fine-tuning a model for a specific task can be expensive if the entire weight matrix is updated. LLMs range from billions to trillions of parameters, making fine-tuning infeasible for many applications.


### Low Rank Decomposition {#low-rank-decomposition}

Low Rank Adaptation (LoRA) is a method of decomposing the weight update matrix \\(\Delta W\\) into smaller matrices \\(A\\) and \\(B\\) such that \\(\Delta W \approx AB\\). The rank \\(r\\) of the decomposition is a hyperparameter that can be tuned to balance performance and computational cost (<a href="#citeproc_bib_item_1">Hu et al. 2021</a>).


### Efficiency {#efficiency}

LoRA is more efficient than fine-tuning because the rank \\(r\\) is much smaller than the number of parameters in the model. This allows for faster training and inference times. If \\(W \in \mathbb{R}^{d \times d}\\), then \\(A \in \mathbb{R}^{d \times r}\\) and \\(B \in \mathbb{R}^{r \times d}\\), so the number of parameters in the decomposition is \\(2dr\\). Compare this to the \\(d^2\\) parameters in the original weight matrix.


### Implementation {#implementation}

When fine-tuning with LoRA, the original weights are frozen. This has a significant impact on the performance of the model since the gradients do not need to be stored for the original weights. Only the gradients for the decomposition matrices \\(A\\) and \\(B\\) need to be stored.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. “LoRA: Low-Rank Adaptation of Large Language Models.” arXiv. <a href="https://doi.org/10.48550/arXiv.2106.09685">https://doi.org/10.48550/arXiv.2106.09685</a>.</div>
</div>
