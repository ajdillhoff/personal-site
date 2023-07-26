+++
title = "Transformers"
authors = ["Alex Dillhoff"]
date = 2022-11-06T00:00:00-05:00
tags = ["deep learning", "llms"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Definition](#definition)
- [Attention](#attention)
- [Key-value Store](#key-value-store)
- [Scaled Dot Product Attention](#scaled-dot-product-attention)
- [Multi-Head Attention](#multi-head-attention)
- [Encoder-Decoder Architecture](#encoder-decoder-architecture)
- [Encoder](#encoder)
- [Decoder](#decoder)
- [Usage](#usage)
- [Resources](#resources)

</div>
<!--endtoc-->



## Introduction {#introduction}

The story of Transformers begins with "Attention Is All You Need" (Vaswani et al., n.d.). In this seminal work, the authors describe the current landscape of sequential models, their shortcomings, and the novel ideas that result in their successful application.

Their first point highlights a fundamental flaw in how [Recurrent Neural Networks]({{< relref "recurrent_neural_networks.md" >}}) process sequential data: their output is a function of the previous time step. Given the hindsight of 2022, where large language models are crossing the [trillion parameter milestone](https://arxiv.org/pdf/2101.03961.pdf), a model requiring recurrent computation dependent on previous time steps without the possibility of parallelization would be virtually intractable.

The second observation refers to attention mechanisms, a useful addition to sequential models that enable long-range dependencies focused on specific contextual information. When added to translation models, attention allows the model to focus on particular words (Bahdanau, Cho, and Bengio 2016).

The Transformer architecture considers the entire sequence using only attention mechanisms.
There are no recurrence computations in the model, allowing for higher efficiency through parallelization.


## Definition {#definition}

The original architecture consists of an encoder and decoder, each containing one or more attention mechanisms.
Not every type of model uses both encoders and decoders. This is discussed later [TODO: discuss model types].
Before diving into the architecture itself, it is important to understand what an attention mechanism is and how it functions.


## Attention {#attention}

Attention mechanisms produce relationships between sequences. When we look at an image of a dog running in a field with the intent of figuring out what the dog is doing in the picture, we pay greater attention to the dog and look at contextual cues in the image that might inform us of their task. This is an automatic process which allows us to efficiently process information.

Attention mechanisms follow the same concept. Consider a machine translation task in which a sentence in English is translated to French. Certain words between the input and output will have stronger correlations than others.


### Soft Attention {#soft-attention}

Use of context vector that is dependent on a sequence of annotations. These contain information about the input sequence with a focus on the parts surrounding the $i$-th word.

\\[
c\_i = \sum\_{j=1}^{T\_x}\alpha\_{ij}h\_j
\\]

What is \\(\alpha\_{ij}\\) and how is it computed? This comes from an alignment model which assigns a score reflecting how well the inputs around position \\(j\\) and output at position \\(i\\) match, given by

\\[
e\_{ij} = a(s\_{i-1}, h\_j),
\\]

where \\(a\\) is a feed-forward neural network and \\(h\_j\\) is an annotation produced by the hidden layer of a BRNN.
These scores are passed to the softmax function so that \\(\alpha\_{ij}\\) represents the weight of annotation \\(h\_j\\):

\\[
\alpha\_{ij} = \frac{\exp(e\_{ij})}{\sum\_{k=1}^{T\_x} \exp (e\_{ik})}.
\\]

This weight reflects how important \\(h\_j\\) is at deciding the next state \\(s\_i\\) and generating \\(y\_i\\).


### Soft vs. Hard Attention {#soft-vs-dot-hard-attention}

This mechanism was also described in the context of visual attention as "soft" attention (Xu et al. 2016).
The authors also describe an alternative version they call "hard" attention.
Instead of providing a probability of where the model should look, hard attention provides a single location that is sampled from a multinoulli distribution parameterized by \\(\alpha\_i\\).

\\[
p(s\_{t,i} = 1 | s\_{j<t}, \mathbf{a}) = \alpha\_{t,i}
\\]

Here, \\(s\_{t,i}\\) represents the location \\(i\\) at time \\(t\\), \\(s\_{j<t}\\) are the location variables prior to \\(t\\), and \\(\mathbf{a}\\) is an image feature vector.

{{< figure src="/ox-hugo/2022-11-10_12-07-42_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Hard attention for \"A man and a woman playing frisbee in a field.\" (Xu et al.)" >}}

{{< figure src="/ox-hugo/2022-11-10_12-08-44_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Soft attention for \"A woman is throwing a frisbee in a park.\" (Xu et al.)" >}}

The two figures above show the difference between soft and hard attention.
Hard attention, while faster at inference time, is non-differentiable and requires more complex methods to train (TODO: cite Luong).


### Self-Attention {#self-attention}

Self attention is particularly useful for determining the relationship between different parts of an input sequence. The figure below demonstrates self-attention given an input sentence (Cheng, Dong, and Lapata 2016).

{{< figure src="/ox-hugo/2022-11-10_13-11-31_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Line thickness indicates stronger self-attention (Cheng et al.)." >}}

-   How aligned the two vectors are.


### Cross Attention {#cross-attention}

TODO


## Key-value Store {#key-value-store}

Query, key, and value come from the same input (self-attention).

Check query against all possible keys in the dictionary. They have the same size.
The value is the result stored there, not necessarily the same size.
Each item in the sequence will generate a query, key, and value.

The attention vector is a function of they keys and the query.

Hidden representation is a function of the values and the attention vector.

The Transformer paper talks about queries, keys, and values. This idea comes from retrieval systems.
If you are searching for something (a video, book, song, etc.), you present a system your query. That system will compare your query against the keys in its database. If there is a key that matches your query, the value is returned.

\\[
att(q, \mathbf{k}, \mathbf{v}) = \sum\_i v\_i f(q, k\_i),
\\]
where \\(f\\) is a similarity function.

This is an interesting and convenient representation of attention.
To implement this idea, we need some measure of similarity.
Why not orthogonality? Two vectors that are orthogonal produce a scalar value of 0.
The maximum value two vectors will produce as a result of the dot product occurs when the two vectors have the exact same direction.
This is convenient because the dot product is simple and efficient and we are already performing these calculations in our deep networks in the form of matrix multiplication.


## Scaled Dot Product Attention {#scaled-dot-product-attention}

{{< figure src="/ox-hugo/2022-11-21_18-39-01_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Scaled dot-product attention ((Vaswani et al., n.d.))" >}}

Each **query** vector is multiplied with each **key** using the dot product.
This is implemented more efficiently via matrix multiplication.
A few other things are added here to control the output.
The first is **scaling**.


## Multi-Head Attention {#multi-head-attention}

A single attention head can transform the input into a single representation. Is this analagous to using a single convolutional filter? The benefit of having multiple filters is to create multiple possible representations from the same input.


## Encoder-Decoder Architecture {#encoder-decoder-architecture}

The original architecture of a transformer was defined in the context of sequence transduction tasks, where both the input and output are sequences. The most common task of this type is machine translation.


## Encoder {#encoder}

The encoder layer takes an input sequence \\(\\{\mathbf{x}\_t\\}\_{t=0}^T\\) and transforms it into another sequence \\(\\{\mathbf{z}\_t\\}\_{t=0}^T\\).

-   What is \\(\mathbf{z}\_t\\)?
-   How is it used?
    Input as key and value into second multi-head attention layer of the **decoder**.
-   Could you create an encoder only model?
    Yes. Suitable for classification tasks -- classify the representation produced by the encoder.
    **How does this representation relate to understanding?**
-   It's a transformation to another representation.

    Generated representation also considers the context of other parts of the same sequence (bi-directional).


## Decoder {#decoder}

-   Generates an output sequence.
-   Decoder-only models?
    Suitable for text generation.
-   What does the input represent?
-   What does the output represent?
-   What if we don't use an encoder, what information is added in lieu of the encoder output?

    <!-- This HTML table template is generated by emacs/table.el -->
    <table border="1">
      <tr>
        <td align="left" valign="top">
          &nbsp;Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
        <td align="left" valign="top">
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Examples&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
        <td align="left" valign="top">
          &nbsp;Tasks&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
      </tr>
      <tr>
        <td align="left" valign="top">
          &nbsp;Encoder&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
        <td align="left" valign="top">
          &nbsp;ALBERT,&nbsp;BERT,&nbsp;DistilBERT,<br />
          ELECTRA,&nbsp;RoBERTa&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
        <td align="left" valign="top">
          &nbsp;Sentence&nbsp;classification,&nbsp;named&nbsp;entity&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />
          recognition,&nbsp;extractive&nbsp;question&nbsp;answering&nbsp;
        </td>
      </tr>
      <tr>
        <td align="left" valign="top">
          &nbsp;Decoder&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
        <td align="left" valign="top">
          &nbsp;CTRL,&nbsp;GPT,&nbsp;GPT-2,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />
          Transformer&nbsp;XL&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
        <td align="left" valign="top">
          &nbsp;Text&nbsp;generation&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
      </tr>
      <tr>
        <td align="left" valign="top">
          &nbsp;Encoder-decoder&nbsp;<br />
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
        <td align="left" valign="top">
          &nbsp;BART,&nbsp;T5,&nbsp;Marian,&nbsp;mBART&nbsp;&nbsp;<br />
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
        <td align="left" valign="top">
          &nbsp;Summarization,&nbsp;translation,&nbsp;generative&nbsp;&nbsp;&nbsp;&nbsp;<br />
          question&nbsp;answering&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        </td>
      </tr>
    </table>


## Usage {#usage}

TODO


## Resources {#resources}

-   <https://twitter.com/labmlai/status/1543159412940242945?s=20&t=EDu5FzDWl92EqnJlWvfAxA>
-   <https://en.wikipedia.org/wiki/Transduction_(machine_learning)>
-   <https://www.apronus.com/math/transformer-language-model-definition>
-   <https://lilianweng.github.io/posts/2018-06-24-attention/>
-   <http://nlp.seas.harvard.edu/annotated-transformer/>
