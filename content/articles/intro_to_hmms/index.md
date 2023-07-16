+++
title = "An Introduction to Hidden Markov Models for Gesture Recognition"
authors = ["Alex Dillhoff"]
date = 2023-07-15T00:00:00-05:00
draft = false
summary = "Hidden Markov Models provide a way of modeling the dynamics of sequential information. They have been used for speech recognition, part-of-speech tagging, machine translation, handwriting recognition, and, as we will see in this article, gesture recognition."
tags = ["machine learning"]
+++

## Introduction {#introduction}

Hidden Markov Models provide a way of modeling the dynamics of sequential information. They have been used for speech recognition, part-of-speech tagging, machine translation, handwriting recognition, and, as we will see in this article, gesture recognition.

Consider a somewhat practical use-case: you are going to throw a party with a meticulously curated playlist. You would rather not let anyone have the remote as it might get lost, and letting anyone interrupt the playlist with their own selections may derail the entire event. However, you still want to give your guests the ability to control the volume and skip back and forth between tracks in the playlist. We will also assume that guests will use change tracks and control the volume responsibly.

The solution to this problem is to implement a gesture recognition system to identify simple hand motions. In this case, we only have to model 4 separate gestures: VolumeUp, VolumeDown, PrevTrack, NextTrack. Since the motions are temporal in nature, we can model each gesture using Hidden Markov Models. First, we need to cover a bit of background on what a Hidden Markov Model actually is.


## Background {#background}

-   First, introduce Markov Chains
-   Then the Markov assumption

At the core of our problem, we want to model a distribution over a sequence of states. Consider a sequence of only 3 states \\(p(x\_1, x\_2, x\_3)\\). The full computation of this can be done using the chain rule of probability:

\\[
p(x\_1, x\_2, x\_3) = p(x\_1) p(x\_2 | x\_1) p(x\_3 | x\_1, x\_2).
\\]

If the random variables of our problem are not conditionally independent, the complexity of calculating this is exponential in the number of random variables.

The **Markov** in Hidden Markov Models addresses this complexity. The **Markov Assumption** states that the probability of an event at time \\(t\\) is conditioned _only_ on the previously observed event: \\(p(x\_t | x\_{t-1})\\). This is compactly represented with a graphical model, as seen in figure **TODO**.

**TODO: Figure of basic Markov Chain**

The **hidden** qualifier comes from the fact that the data we wish to model was generated from some underlying process that is not directly observable. A classic example for HMMs uses the weather. Imagine you had a log which had the number of water bottles a person had drank per day over the entire year. To make the problem slightly more difficult, the log entries were not associated with a date. It is reasonable to say that the amount of water a person drinks is influenced by how hot or cold it is on a particular day. So, the **hidden state** in this case is the weather: hot or cold. We can model this with an HMM by establishing that the amount of water (**observed state**) is conditioned on the weather (**hidden state**). Figure **TODO** shows this HMM graphically.

**TODO: Figure of HMM**

Formally, a Hidden Markov Model is defined by

-   The number of hidden states \\(N\\).
-   A transition probability matrix \\(A \in \mathbb{R}^{N \times N}\\), where \\(a\_{ij} = p(z\_t = j | z\_{t-1} = i)\\).
-   An observation symbol probability distribution \\(B = \\{b\_j(k)\\} = p(\mathbf{x}\_t = k | z\_t = j)\\).
-   An initial state distribution \\(\pi\_i = p(z\_t = i)\\).

The trainable parameters of our model are \\(\lambda = (A, B, \pi)\\).


## Functions of an HMM {#functions-of-an-hmm}

Given the basic definition of what an HMM is, how can we train the parameters defined in \\(\lambda\\). If we somehow already knew the parameters, how can we extract useful information from the model? Depending on our task, we can use HMMs to answer many important questions:

-   **Filtering** computes \\(p(z\_t | \mathbf{x}\_{1:t})\\). That is, we are computing this probability as new samples come in up to time \\(t\\).
-   **Smoothing** is accomplished when we have all the data in the sequence.
    This is expressed as \\(p(z\_t|\mathbf{x}\_{1:T})\\).
-   **Fixed lag smoothing** allows for a trade off between accuracy and delay. It is useful in cases where we might not have the full sequence, but we wish to compute \\(p(z\_{t-l}|\mathbf{x}\_{1:t})\\) for some \\(l > 0\\).
-   **Predictions** are represented as \\(p(z\_{t+h}|\mathbf{x}\_{1:t})\\), where \\(h > 0\\).
-   **MAP estimation** yields the most probably state sequence \\(\text{arg}\max\_{\mathbf{z}\_{1:T}}p(\mathbf{z}\_{1:T}|\mathbf{x}\_{1:T})\\).
-   We can sample the **posterior** \\(p(\mathbf{z}\_{1:T}|\mathbf{x}\_{1:T})\\).
-   We can also compute \\(p(\mathbf{x}\_{1:T})\\) by summing up over all hidden paths. This is useful for classification tasks.

Of course not all of these functions make sense for every possible task, more on that later. This article is not meant to be an exhaustive resource for all HMM functions; we will only look at the tasks necessary to train and use HMMs for isolated gesture recognition **TODO: offer additional reading suggestions**.


## Data Processing {#data-processing}

As far as the efficacy of our model goes, how we process the data is the most important. Our system will start with a camera that records our guests performing one of the four simple motions. For simplicity, let's pretend that the camera has an onboard chip that detects the 2D centroids of the left hand for each frame. That helps a lot, but there is still the problem of isolating a group of frames based on when the user wanted to start and finish the command. Assuming we have a solution for both of these problems, we still need to take into account that users will gesture at different speeds. Since all of these problems are challenging in their own right, we will assume the computer vision fairy has taken care of this for us.

Each gesture in our dataset consists of 30 \\((x, y)\\) locations of the center of the left hand with respect to image coordinates. Even with this simplified data, we have another problem: different users may gesture from different locations. The hand locations for one user performing the `VolumeUp` gesture may be vastly different from another. This isn't too bad to deal with. We could normalize or training data by subtracting the location of the hand in the first frame from the gesture. That way every input would start at \\((0, 0)\\). We can simplify this even further by using **relative motion states**.


### Relative Motion States {#relative-motion-states}

Relative motion states discretize our data, thus simplifying the input space. The idea is quite simple: if the hand moved to the right relative to the previous frame, we assign \\(x = 1\\) for that frame. If it moved to the left, assign \\(x = -1\\). If it didn't move at all, or did not move a significant amount, assign \\(x = 0\\). We apply similar rules for the \\(y\\) locations as well. The **TODO: figure** below shows the relative motion grid.

Besides greatly simplifying our input space, meaning we can use a simple categorical distribution to model these observations, we no longer have to worry about the discrepency between where each user performed the gesture.


## Modeling a Gesture {#modeling-a-gesture}

Our system will consist of 4 HMM models to model the dynamics of each gesture. To determine which gesture was performed, we will given our input sequence to each one and have it compute \\(p(\mathbf{x}\_{1:T}; \lambda\_i)\\), the probability of the observation given the parameters of model \\(i\\). Whichever model gives the high probability wins.

**TODO**

1.  Describe EM at a high level, show the breakdown of probabilities that need to be known
2.  Go into forward-backwards
3.  Go back to EM and plug them in


### Training: Expectation-Maximization {#training-expectation-maximization}

If we cannot observe the hidden states directly, how are we supposed to update the model parameters \\(\lambda = (A, B, \pi)\\)? We may not have all of the information, but we do have _some_ information. We can use that to fill in the missing values with what we would expect them to be given what we already know. Then, we can update our parameters using those expected values. This is accomplished through a two-stage algorithm called **Expectation-Maximization**. Those familiar with k-Nearest Neighbors should already be familiar with this process.


#### Updating with Perfect Information {#updating-with-perfect-information}

It is useful to know how we would update our parameters assuming we had perfect information. If the hidden states were fully observable, then updating our model parameters would be as straightforward as computing the maximum likelihood estimates.
For \\(A\\) and \\(\pi\\), we first tally up the following counts:

\\[
\hat{a}\_{ij} = \frac{N\_{ij}}{\sum\_j N\_{ij}},
\\]

the number of times we expect to transition from \\(i\\) to \\(j\\) divided by the number of times we transition from \\(i\\) to any other state. Put simply, this computes the expected transitions from \\(i\\) to \\(j\\) normalized by all the times we expect to start in state \\(i\\).

For \\(\pi\\), we have

\\[
\hat{\pi\_i} = \frac{N\_i}{\sum\_i N\_i},
\\]

the number of times we expect to start in state \\(i\\) divided by the number of times we start in any other state.

Estimating the parameters for \\(B\\) depends on which distribution we are using for our observation probabilities.
For a multinomial distribution, we would compute the number of times we are in state \\(j\\) and observe a symbol \\(k\\) divided by the number of times we are in state \\(j\\):

\\[
\hat{b}\_{jk} = \frac{N\_{jk}}{N\_k},
\\]

where

\\[
N\_{jk} = \sum\_{i=1}^N \sum\_{t=1}^T \mathbb{1} (z\_{i, t}=j, x\_{i, t}=k).
\\]

It is also common to model our emission probability using a Normal distribution. We can even use a parameterized model like a neural network. **TODO: provide links to examples of these**


#### Updating with Missing Information {#updating-with-missing-information}

Now to the real problem: fill in our missing information using our observable data and the current parameter estimates. There are two important statistics that we need to compute, called the **sufficient statistics**.

1.  The expected number of transitions from \\(i\\) to \\(j\\).
2.  The expected number of times we are transitioning from \\(i\\) to any other state.

Both of these can be computed starting with the same probability _conditioned_ on our observable data:

\\[
p(z\_t = i, z\_{t+1} = j|\mathbf{x}\_{1:T}).
\\]


### Forwards-Backwards Algorithm {#forwards-backwards-algorithm}


## Implementation in Python {#implementation-in-python}


## Conclusion {#conclusion}
