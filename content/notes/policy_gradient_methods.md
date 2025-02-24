+++
title = "Policy Gradient Methods"
authors = ["Alex Dillhoff"]
date = 2025-02-18T00:00:00-06:00
tags = ["reinforcement learning"]
draft = false
sections = "Machine Learning"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Policy Gradients](#policy-gradients)

</div>
<!--endtoc-->

When we had full knowledge of the states, we could use [Markov Decision Processes]({{< relref "markov_decision_processes.md" >}}) to find the optimal policy. When this assumption breaks down, we need to come up with our best approximation. This is not a far stretch from how we might handle new scenarios in our own lives. When we begin a new task, we are certainly not experts. We may learn from a teacher or set off to explore on our own. As we practice and churn out the seemingly endless variations of our endeavour, we begin to develop a sense of what works and what doesn't. We may not be able to articulate the exact rules that we follow, but we can certainly tell when we are doing well or poorly.

In lieu of a conscious agent with human intelligence, we can approximate the policy using a gradient-based approach. We will use the gradient of the expected reward with respect to the policy parameters to update the policy. Methods that use this approach are called ****policy gradient methods****.


## Policy Gradients {#policy-gradients}

Q-Learning is an _off-policy_ learning algorithm that directly estimates optimal action values in order to find optimal policies. That is, it does not update a policy table or function. Instead, it chooses the best action given the computed estimates of the action values. Given a state-action space that is too complex to be efficiently represented as a table, the function must be approximated. This can be done by taking the derivative of the cost fnction with respect to its objective function.

Whereas value function methods like Q-Learning approximate a value function, **policy gradient methods** approximate the policy function. Given this estimate, the gradient is updated to maximize the expected reward.

\\[
\theta\_{t+1} = \theta\_t + \alpha \nabla J(\theta)
\\]


### Episodic Case {#episodic-case}

For episodic problems, we can use the **average state value**

\\[
\bar{v}\_{\pi} = \sum\_{s} d\_{\pi}(s) v\_{\pi}(s) = \sum\_{s} d\_{\pi}(s) \sum\_{a} \pi\_{\theta}(a|s) q\_{\pi}(s, a),
\\]

where \\(d\_{\pi}(s)\\) is the stationary distribution of states under the policy \\(\pi\\) and \\(q\_{\pi}(s, a)\\) is the action-value function. The fact that \\(d\_{\pi}(s)\\) is stationary is very important when computing the gradients.


#### What is a Stationary Distribution? {#what-is-a-stationary-distribution}

If \\(\pi\\) is a stationary distribution and \\(P\\) is the transition matrix over a Markov chain, then \\(\pi\\) satisfies the equation \\(\pi = \pi P\\). This means that the distribution of states under the policy \\(\pi\\) does not change over time.


#### Significance of the Stationary Distribution {#significance-of-the-stationary-distribution}

The stationary distribution simplifies the computation of the gradient. It allows us to avoid computing how the state distributions change with respect to the policy parameters. That is because the particular trajectory our agent takes does not affect the probability of being in any given state.


#### Gradient of the Episodic Case {#gradient-of-the-episodic-case}

The gradient of the average state value is

\\[
\nabla\_{\theta} J(\theta) = \mathbb{E}\_{\pi} \left[ \nabla\_{\theta} \log \pi\_{\theta}(a|s) q\_{\pi}(s, a) \right].
\\]

An important note here is that this a stochastic gradient. We surely do not have complete knowledge of the environment, so computing the true gradient is out of the question. This also means that our stochastic gradients may exhibity high variance due to the randomness of the environment and policy.


### Continuous Case {#continuous-case}

Using the average value in this case would not directly align with this goal because the value function represents the expected rewards starting from a specific state. In continuing problems, we are interested in global performance across _all_ states under the policy.
