+++
title = "Pretraining Large Language Models"
authors = ["Alex Dillhoff"]
date = 2023-11-16T00:00:00-06:00
tags = ["large language models", "machine learning"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Unsupervised Pre-training](#unsupervised-pre-training)
- [From GPT to GPT2](#from-gpt-to-gpt2)

</div>
<!--endtoc-->

These notes provide an overview of pre-training large language models like GPT and Llama.


## Unsupervised Pre-training {#unsupervised-pre-training}

Let's start by reviewing the pre-training procedure detailed in the GPT paper (<a href="#citeproc_bib_item_1">Radford et al. 2020</a>). The _Generative_ in Generative Pre-Training reveals much about how the network can be trained without direct supervision. It is analogous to how you might have studied definitions as a kid: create some flash cards with the term on the front and the definition on the back. Given the context of the word, you try and recite the definition. For a pre-training language model, it is given a series of tokens and is tasked with generating the next token in the sequence. Since we have access to the original documents, we can easily determine if it was correct.

Given a sequence of tokens \\(\mathcal{X} = \\{x\_1, x\_2, \ldots, x\_n\\}\\), the model is trained to predict the next token \\(x\_{n+1}\\) in the sequence. The model is trained to maximize the log-likelihood of the next token:

\\[\mathcal{L}(\mathcal{X}) = \sum\_{i=1}^{n} \log p(x\_{i+1} \mid x\_{i-k}, \ldots, x\_i)\\]

where \\(k\\) is the size of the context window.

Large language models are typically based on the [Transformers]({{< relref "transformers.md" >}}) model. The original model was trained for language translation. Depending on the task, different variants are employed. For GPT models, a decoder-only architecture is used, as see below.

{{< figure src="/ox-hugo/2023-11-16_15-56-14_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Decoder-only diagram from (<a href=\"#citeproc_bib_item_2\">Vaswani et al. 2017</a>)." >}}

The entire input pipeline for GPT can be expressed rather simply. First, the tokenized input is passed through an embedding layer \\(W\_{e}\\). Embedding layers map the tokenized input into a lower-dimensional vector representation. A positional embedding matrix of the same size as \\(\mathcal{X} W\_{e}\\) is added in order to preserve the order of the tokens.

The embedded data \\(h\_0\\) is then passed through \\(n\\) transformer blocks. The output of this is passed through the softmax function in order to produce an output distribution over target tokens.


## From GPT to GPT2 {#from-gpt-to-gpt2}

GPT2 is a larger version of GPT, with an increased context size of 1024 tokens and a vocabulary of 50,257 vocabulary. In this paper, they posit that a system should be able to perform many tasks on the same input. For example, we may want our models to summarize complex texts as well as provide answers to specific questions we have about the content. Instead of training multiple separate models to perform these tasks individually, the model should be able to adapt to these tasks based on the context. In short, it should model \\(p(output \mid input, task)\\) instead of \\(p(output \mid input)\\).

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Radford, Alec, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2020. “Improving Language Understanding by Generative Pre-Training,” 12.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. “Attention Is All You Need,” 11.</div>
</div>
