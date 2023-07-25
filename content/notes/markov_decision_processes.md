+++
title = "Markov Decision Processes"
authors = ["Alex Dillhoff"]
date = 2023-07-24T00:00:00-05:00
draft = false
tags = ["reinforcement learning"]
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Key Terms](#key-terms)
- [Policies and Values](#policies-and-values)
- [Value Iteration Algorithm](#value-iteration-algorithm)
- [Policy Iteration](#policy-iteration)

</div>
<!--endtoc-->



## Key Terms {#key-terms}

-   **Agent**: The learner or decision maker.
-   **Environment**: The world that the agent can interact with.
-   **State**: A representation of the agent and environment.
-   **Action**: The agent can take an action in the environment.
-   **Reward**: Given to the agent based on actions taken.

**Goal**: Maximize rewards earned over time.

At time \\(t\\), the agent observes the state of the environment \\(S\_t \in \mathcal{S}\\) and can select an action \\(A\_t \in \mathcal{A}(s)\\), where \\(\mathcal{A}(s)\\) suggests that the available actions are dependent on the current state.
At time \\(t + 1\\), the agent receives a reward \\(R\_{t+1} \in \mathcal{R}\\).

{{< figure src="/ox-hugo/2022-11-15_19-01-30_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>The agent-environment in a Markov decision process (Credit: Sutton &amp; Barto)." >}}

To improve its knowledge about an environment or increase its performance on a task, an agent must first be able to interpret or make sense of that environment in some way.
Second, there must be a well defined goal. For an agent playing Super Mario, for example, the goal would be to complete each level while maximizing the score.
Third, the agent must be able to interact with its environment by taking actions.
If the Mario-playing agent could not move Mario around, it would never be able to improve.
If the agent makes a decision which leads to Mario's untimely demise, it would update its knowledge of the world so that it would tend towards a more favorable action.
These three requirements: sensations, actions, and goals, are encapsulated by **Markov Decision Processes**.

A Markov decision process is defined by

-   \\(\mathcal{S}\\) - a set of states,
-   \\(\mathcal{A}\\) - a set of actions,
-   \\(\mathcal{R}\\) - a set of rewards,
-   \\(P\\) - the transition probability function to determine transition between states,
-   \\(\gamma\\) - discount factor for future rewards.

At time \\(t\\), an agent in state \\(S\_t\\) selects an action \\(A\_t\\).
At \\(t+1\\), it receives a reward \\(R\_{t+1}\\) based on that action.

In a finite MDP, the states, actions, and rewards have a finite number of elements.
Random variables \\(R\_t\\) and \\(S\_t\\) have discrete probability distributions dependent on the preceding state and action.

\\[
p(s', r|s, a) = P\\{S\_t = s', R\_t = r|S\_{t-1} = s, A\_{t-1} = a\\}
\\]

If we want the state transition probabilities, we can sum over the above distribution:

\\[
p(s'|s, a) = P\\{S\_t = s'|S\_{t-1} = s, A\_{t-1}=a\\} = \sum\_{r\in\mathcal{R}}p(s', r|s, a).
\\]


### Defining Goals {#defining-goals}

In reinforcement learning, the goal is encoded in the form of a ****reward signal****. The agent sets out to _maximize_ the total amount of reward it receives over an ****episode****. An ****episode**** is defined dependent on the problem context and ends in a ****terminal state****. It could be a round of game, a single play, or the result of moving a robot. Typically, the rewards come as a single scalar value at teach time step. This implies that an agent might take an action that results in a negative reward if it is optimal in the long run.

Formally, the **expected return** includes a **discount factor** that allows us to control the trade-off between short-term and long-term rewards:

\\[
G\_t = \sum\_{k=0}^{\infty} \gamma^k R\_{t+k+1},
\\]

where \\(0 \leq \gamma \leq 1\\). This can be written in terms of the expected return itself as well:

\\[
G\_t = R\_{t+1} + \gamma G\_{t+1}.
\\]


## Policies and Values {#policies-and-values}

Two important concepts that help our agent make decisions are the policy and value functions. A **policy**, typically denoted by \\(\pi\\), maps states to actions. Such a function can be deterministic, \\(\pi(s) = a\\), or stochastic, \\(\pi(a|s)\\).

The value of a particular state under a policy \\(\pi\\) is defined as

\\[
v\_{\pi}(s) = \mathbb{E}\_{\pi}[G\_t | S\_t = s] = \mathbb{E}\_{\pi}\Bigg[\sum\_{k=0}^{\infty}\gamma^k R\_{t+k+1}\Bigg|S\_t=s\Bigg].
\\]

We also must define the value of taking an action \\(a\\) in state \\(s\\) following policy \\(\pi\\):

\\[
q\_{\pi}(s, a) = \mathbb{E}\_{\pi}[G\_t|S\_t=s, A\_t=a] = \mathbb{E}\_{\pi}\Bigg[\sum\_{k=0}^{\infty}\gamma^k R\_{t+k+1}\Bigg|S\_t=s, A\_t=a\Bigg].
\\]

This function defines the expected return of following a particular policy and starting in state \\(s\\).
Both the **state-value** and **action-value** functions can be updated as a result of the agent's experience. How it is updated is method-dependent. Certain methods will also dictate how the policy itself can be updated.


## Value Iteration Algorithm {#value-iteration-algorithm}

**Bellman Equation**

\\[
U(s) = R(s) + \gamma \max\_{a \in A(s)}\Big\\{\sum\_{s'}p(s'|s, a)U(s')\Big\\}
\\]

Take an action \\(a\\) and compute \\(\sum\_{s'}p(s'|s, a)U(s')\\) -- the expected utility of all possible states accessible from the current state (assuming you perform action \\(a\\)). Then, consider the same thing for all possible actions the agent can take.


## Policy Iteration {#policy-iteration}
