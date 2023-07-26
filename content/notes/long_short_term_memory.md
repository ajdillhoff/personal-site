+++
title = "Long Short-Term Memory"
authors = ["Alex Dillhoff"]
date = 2022-04-12T00:00:00-05:00
tags = ["deep learning"]
draft = false
+++

The recurrent nature of RNNs means that gradients get smaller and smaller as the timesteps increase.
This is known as the **vanishing gradient problem**.
One of the first popular solutions to this problem is called **Long Short-Term Memory**, a recurrent network architecture by Hochreiter and Schmidhuber.

An LSTM is made up of memory blocks as opposed to simple hidden units.
Each block is differentiable and contains a memory cell along with 3 gates: the input, output, and forget gates.
These components allow the blocks to maintain some history of information over longer range dependencies.

{{< figure src="/ox-hugo/2022-04-14_13-36-14_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>LSTM memory block with a single cell (adapted from Andrew Ng's diagram)." >}}

The original LSTM only had an input and output gate.
Forget gates were added in 2000 by Gers et al. to control the amount of context that could be reset, if the task called for it.
Peephole connections were proposed by Gers et al. in 2002.
These are weights that combine the previous cell state to the gates in order to learn tasks that require precise timing.

By controlling when information can either enter or leave the memory cell, LSTM blocks are able to maintain more historical context than RNNs.


## Forward Pass {#forward-pass}

The equations listed here follow notation and description from Alex Graves' thesis.

The weight from unit \\(i\\) to \\(j\\) is given as \\(w\_{ij}\\).
The input to unit \\(j\\) at time \\(t\\) is \\(a\_j^t\\) and the result of its activation function is \\(b\_j^t\\).
Let \\(\psi\\), \\(\phi\\), and \\(\omega\\) be the input, forget, and output gates.
A memory cell is denoted by \\(c \in C\\), where \\(C\\) is the set of cells in the network.
The activation, or state, of a given cell \\(c\\) at time \\(t\\) is \\(s\_c^t\\).
The output of each gate passes through an activation function \\(f\\), while the input and output activation functions of a memory block are given by \\(g\\) and \\(h\\).

The forward pass for the input gates is

\\[
a\_{\psi}^t = \sum\_{i=1}^I w\_{i\psi}x\_i^t + \sum\_{h=1}^H w\_{h\psi}b\_h^{t-1} + \sum\_{c=1}^C w\_{c\psi}s\_c^{t-1}.
\\]

The output of the forget gates is

\\[
a\_{\phi}^t = \sum\_{i=1}^I w\_{i\phi}x\_i^t + \sum\_{h=1}^H w\_{h\phi}b\_h^{t-1} + \sum\_{c=1}^C w\_{c\phi}s\_c^{t-1}.
\\]

The output of the output gates is

\\[
a\_{\omega}^t = \sum\_{i=1}^I w\_{i\omega}x\_i^t + \sum\_{h=1}^H w\_{h\omega}b\_h^{t-1} + \sum\_{c=1}^C w\_{c\omega}s\_c^{t-1}.
\\]

Each of the outputs above is passed through an activation function \\(f\\).

The output of each cell is computed as

\\[
a\_c^t = \sum\_{i=1}^I w\_{ic}x\_i^t + \sum\_{i=1}^H w\_{hc}b\_h^{t-1}
\\]

and the internal state is updated via

\\[
s\_c^t = b\_{\phi}^t s\_c^{t-1} + b\_{\psi}^t g(a\_c^t).
\\]

The state update considers the state at the previous timestep multiplied by the output of the forget gate.
That is, it controls how much of the current memory to keep.

The final cell output is given as

\\[
b\_c^t = b\_{\omega}^t h(s\_c^t).
\\]
