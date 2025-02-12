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
- [Translation and Rotation](#translation-and-rotation)
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

The positional encoding used in the original Transformer architecture is called **sinusoidal positional encoding** (<a href="#citeproc_bib_item_3">Vaswani et al. 2017</a>). Given the position \\(pos\\) and dimension \\(i\\) of the input, the encoding is given by

\begin{align\*}
PE\_{pos, 2i} &= \sin\left(\frac{pos}{10000^{2i/d\_{model}}}\right)\\\\
PE\_{pos, 2i+1} &= \cos\left(\frac{pos}{10000^{2i/d\_{model}}}\right)
\end{align\*}

**Why do we use \\(\sin\\) and \\(\cos\\) here?** To understand this, we need to return to trigonometry. I also like to go through the following derivation when discussing geometric linear transformations in my linear algebra class. The second property stated above was that the positions between two elements should be linear. For this, let's ignore the frequency term and only consider the position term.

\begin{align\*}
PE\_{pos} &= \sin(p)\\\\
PE\_{pos+1} &= \cos(p)\\\\
\end{align\*}

Adding some offset of \\(k\\) to the position \\(p\\) gives

\begin{align\*}
PE\_{pos+k} &= \sin(p+k)\\\\
PE\_{pos+1+k} &= \cos(p+k)\\\\
\end{align\*}

Using the trigonometric identity \\(\sin(a+b) = \sin(a)\cos(b) + \cos(a)\sin(b)\\), we can rewrite the above as

\begin{align\*}
PE\_{pos+k} &= \sin(p)\cos(k) + \cos(p)\sin(k)\\\\
PE\_{pos+1+k} &= \cos(p)\cos(k) - \sin(p)\sin(k)\\\\
\end{align\*}

A transformation that can be represented as a matrix multiplication is a linear transformation. So we are looking for a matrix \\(M\\) such that

\begin{align\*}
\begin{bmatrix}
m\_{11} & m\_{12}\\\\
m\_{21} & m\_{22}
\end{bmatrix}
\begin{bmatrix}
\sin(p)\\\\
\cos(p)
\end{bmatrix} &= \begin{bmatrix}
\sin(p)\cos(k) + \cos(p)\sin(k)\\\\
\cos(p)\cos(k) - \sin(p)\sin(k)
\end{bmatrix}
\end{align\*}

If you know how to multiply matrices, you can see that the matrix

\begin{align\*}
\begin{bmatrix}
\cos(k) & \sin(k)\\\\
-\sin(k) & \cos(k)
\end{bmatrix}
\end{align\*}

does the trick. This is a rotation matrix, and it is a linear transformation.


## Translation and Rotation {#translation-and-rotation}

An embedded token is special. The combination of values across the embedding dimensions define what the token represents semantically. If we add to those values, we are fundamentally changing the semantic meaning of the token. The sinusoidal positional encoding does exactly that.

{{< figure src="/ox-hugo/2025-02-12_14-54-34_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Positional encoding is added to the embedding (<a href=\"#citeproc_bib_item_3\">Vaswani et al. 2017</a>)." >}}

As an alternative, we could rotate the query and key vectors in the self-attention mechanism. This would preserve the _magnitude_ of the token's representation while only changing the relative angle between them. This is the approach taken by the **Rotary Position Embedding** (<a href="#citeproc_bib_item_2">Su et al. 2023</a>).


## Rotary Position Embedding (RoPE) {#rotary-position-embedding--rope}

RoPE proposes that the _relative_ positional information be encoded in the product of the query and key vectors of the self-attention mechanism. This is expressed by equation (11) in the original paper (<a href="#citeproc_bib_item_2">Su et al. 2023</a>).

\\[
\langle f\_q(\mathbf{x}\_m, m), f\_k(\mathbf{x}\_n, n)\rangle = g(\mathbf{x}\_m, \mathbf{x}\_n, m - n)
\\]

The derivation of their solution is a bit involved, but key idea is to rotate the transformed embeddings by an angle that is a multiple of the position index:

\begin{equation\*}
f\_{\\{q, k\\}}(\mathbf{x}\_m, m) =
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta)\\\\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\begin{bmatrix}
W\_{\\{q, k\\}}^{(11)} & W\_{\\{q, k\\}}^{(12)}\\\\
W\_{\\{q, k\\}}^{(21)} & W\_{\\{q, k\\}}^{(22)}
\end{bmatrix}
\begin{bmatrix}
\mathbf{x}\_m^{(1)}\\\\
\mathbf{x}\_m^{(2)}
\end{bmatrix}
\end{equation\*}

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Gehring, Jonas, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. 2017. “Convolutional Sequence to Sequence Learning.” arXiv. <a href="https://doi.org/10.48550/arXiv.1705.03122">https://doi.org/10.48550/arXiv.1705.03122</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Su, Jianlin, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2023. “RoFormer: Enhanced Transformer with Rotary Position Embedding.” arXiv. <a href="https://doi.org/10.48550/arXiv.2104.09864">https://doi.org/10.48550/arXiv.2104.09864</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_3"></a>Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. “Attention Is All You Need,” 11.</div>
</div>
