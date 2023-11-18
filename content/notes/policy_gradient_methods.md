+++
title = "Policy Gradient Methods"
authors = ["Alex Dillhoff"]
date = 2023-11-12T00:00:00-06:00
tags = ["reinforcement learning"]
draft = false
+++

When we had full knowledge of the states, we could use [Markov Decision Processes]({{< relref "markov_decision_processes.md" >}}) to find the optimal policy. When this assumption breaks down, we need to come up with our best approximation. This is not a far stretch from how we might handle new scenarios in our own lives. When we begin a new task, we are certainly not experts. We may learn from a teacher or set off to explore on our own. As we practice and churn out the seemingly endless variations of our endeavour, we begin to develop a sense of what works and what doesn't. We may not be able to articulate the exact rules that we follow, but we can certainly tell when we are doing well or poorly.

In lieu of a conscious agent with human intelligence, we can approximate the policy using a gradient-based approach. We will use the gradient of the expected reward with respect to the policy parameters to update the policy. Methods that use this approach are called ****policy gradient methods****.
