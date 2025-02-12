+++
title = "Positional Encoding"
authors = ["Alex Dillhoff"]
date = 2025-02-11T09:12:00-06:00
draft = false
sections = "Machine Learning"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [What are positional encodings?](#what-are-positional-encodings)
- [Why are positional encodings needed?](#why-are-positional-encodings-needed)
- [Properties of Positional Encodings](#properties-of-positional-encodings)
- [Sinusoidal Positional Encoding](#sinusoidal-positional-encoding)
- [Rotary Position Embedding (RoPE)](#rotary-position-embedding--rope)

</div>
<!--endtoc-->



## What are positional encodings? {#what-are-positional-encodings}

Positional encodings are a way to encode the position of elements in a sequence. They are used in the context of sequence-to-sequence models, such as transformers, to provide the model with information about the order of elements in the input sequence. These were not needed for models like RNNs and LSTMs since the order of elements in the input sequence is preserved by the recurrent connections.


## Why are positional encodings needed? {#why-are-positional-encodings-needed}

Imagine your favorite movie. It is highly likely that the characters evolve over time through their experiences with each other and the environment around them. Your perception of your favorite character is almost certainly influenced by these events. If there was no story, no context, then your summary of them would probably not change from one moment to the next. This is what it would be like for a transformer model with no positional encodings. The representation of each input element would be the same regardless of its position in the sequence, even if the context of the surrounding elements is different.


## Properties of Positional Encodings {#properties-of-positional-encodings}

Positional encodings have been used before in the context of convolutional sequence-to-sequence models (<a href="#citeproc_bib_item_1">Gehring et al. 2017</a>). I am focusing on three primary properties, but interested readers should check out [this great article](https://huggingface.co/blog/designing-positional-encoding) by Christopher Fleetwood.


#### Encodings should be unique {#encodings-should-be-unique}

The value given for each position should be unique so that input values can be distinguished. It should also be consistent across different sequences lengths. That is, the encoding for the first element in a sequence of length 10 should be the same as the encoding for the first element in a sequence of length 100.


#### Linearity of position {#linearity-of-position}

The positional encoding should be linear with respect to the position of the element in the sequence. This means that the encoding for the first element should be the same distance from the encoding for the second element as the encoding for the second element is from the encoding for the third element.


#### Generalization to out of training sequence lengths {#generalization-to-out-of-training-sequence-lengths}

The positional encoding should generalize to sequences of different lengths than those seen during training. This is important because the model should be able to handle sequences of different lengths during inference.


## Sinusoidal Positional Encoding {#sinusoidal-positional-encoding}

The positional encoding used in the original Transformer architecture is called **sinusoidal positional encoding** (<a href="#citeproc_bib_item_2">Vaswani et al. 2017</a>). Given the position \\(pos\\) and dimension \\(i\\) of the input, the encoding is given by

\begin{align\*}
PE\_{pos, 2i} &= \sin\left(\frac{pos}{10000^{2i/d\_{model}}}\right)\\\\
PE\_{pos, 2i+1} &= \cos\left(\frac{pos}{10000^{2i/d\_{model}}}\right)
\end{align\*}


## Rotary Position Embedding (RoPE) {#rotary-position-embedding--rope}

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Gehring, Jonas, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. 2017. “Convolutional Sequence to Sequence Learning.” arXiv. <a href="https://doi.org/10.48550/arXiv.1705.03122">https://doi.org/10.48550/arXiv.1705.03122</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. “Attention Is All You Need,” 11.</div>
</div>
