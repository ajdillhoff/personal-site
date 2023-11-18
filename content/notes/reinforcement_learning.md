+++
title = "Reinforcement Learning"
author = ["Alex Dillhoff"]
date = 2023-07-12T00:00:00-05:00
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Topics](#topics)
- [Introduction](#introduction)
- [Definition](#definition)
- [Markov Decision Processes](#markov-decision-processes)
- [RL vs. MDP](#rl-vs-dot-mdp)
- [Passive RL](#passive-rl)
- [Resources](#resources)

</div>
<!--endtoc-->



## Topics {#topics}

-   What is reinforcement learning?
-   Examples
-   Finite Markov Decision Processes
-   Passive vs. Active Methods
-   Adaptive Dynamic Programming
-   Monte Carlo Methods
-   Temporal-Different Learning
-   Q-Learning
-   Function Approximation
-   Deep Q-Learning
-   Policy and Value Iteration
-   Case Studies


## Introduction {#introduction}

When placed in a new environment, we orient and learn about it by interacting with it.
When given a novel task, we may not be able to explicitly describe the task or even perform well at it, but we can learn a lot about it through trial and error.

Image if you were to be thrown into an alien world teaming with unknown life forms.
You would not be able to identify these life forms, but you would be able to learn about their behaviors, shape, surface anatomy, and other attributes based on perception alone.
Simply learning about the structure of the environment is the task of unsupervised learning.

In supervised machine learning, there is typically some objective function that is minimized based on ground truth or target variables that are known ahead of time.
In settings like the ones depicted above, there is no form of supervision.
Instead, a representation of the environment is learned from one's experience and sensory input.

Reinforcement learning maps environment input to actions.
For example, an agent trained to play Chess will evaluate the current board and make a decision based on experience.
Whether or not that decision was beneficial may not immediately be known.
If it results in winning against an opponent, the model would strengthen its knowledge of that particular play.
The concept of trial-and-error and the fact that an agent may not know if the correct decision was made until later are two of the most important concepts of reinforcement learning.

There are many challenges present in reinforcement learning.
The agents consider the entire environment and attempt to maximize some reward value based on a well defined goal.
There is then a trade off between **exploiting** actions that are known to give a positive reward and **exploring** new actions that may lead to a better payoff.

Reinforcement learning is the study of goal-seeking agents that can sense some aspect of their environment and can perform actions that directly affect that environment. An agent is not always a robot, although it is a popular use case. Taking from recent popularity, agents such as [AlphaGo](<https://www.deepmind.com/research/highlighted-research/alphago>) exist within a game playing environment where their goal is win.


### Examples {#examples}

**Gaming:** Agents for traditional board games such as Chess and [Go](<https://www.deepmind.com/research/highlighted-research/alphago>) have been implemented, surpassing even the best players in the world. Video games are also a popular environment for constructing complex agents. Developers can use RL to construct sophisticated characters in the game that behave in complex way as well as react more realistically towards the player. DeepMind introduced a powerful agent for [StarCaft II](<https://www.deepmind.com/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii>) which beat some of the best human players in the world. OpenAI debuted [OpenAI Five](<https://openai.com/research/openai-five-defeats-dota-2-world-champions>) at the 2019 Dota 2 World Championships, defeating the championship team back-to-back.

OpenAI released OpenAI gym, now an open source project named [Gymnasium](<https://github.com/Farama-Foundation/Gymnasium>), which is a library for developing and comparing reinforcement learning algorithms. One of recent interest is [RLGym](<https://www.twitch.tv/rlgym>), a Rocket League agent.


## Definition {#definition}

A reinforcement learning system is identified by the **environment** and an **agent** acting in that environment.
An agent's behavior in the environment is defined by a **policy**.
A policy usually describes a set of rules in response to the current state of the environment.

In order to improve on some task, a **reward signal** is defined.
Over time, an agent should maximize the reward.
In general, an action leading to a favorable outcome should present a high reward value.

A **value function** describes the estimated total reward over the long run given the agent's current state.
Taking an action that results in an immediate reward may lead to a lower payoff in the long run.
This can be predicted from the value function.

Additionally, a **model** of the environment will include prior knowledge about that environment.
This allows the agent to act optimally over a longer sequence of states.


## Markov Decision Processes {#markov-decision-processes}


## RL vs. MDP {#rl-vs-dot-mdp}

Reinforcement learning does not assume the **transition model** or the **reward function**.


## Passive RL {#passive-rl}

The policy is fixed while the transition model and reward function are learned over time.
The goal is to compute the utility of each state.


### Adaptive Dynamic Programming {#adaptive-dynamic-programming}

\\(R(s)\\) and \\(p(s'|s,a)\\) can be updated at each step based solely on the observations.
Using these observations, the utility of each state can be estimated following policy evaluation (TODO: link algorithm for policy evaluation).


## Resources {#resources}

-   <https://web.stanford.edu/class/cs234/modules.html>
-   <https://lilianweng.github.io/posts/2018-02-19-rl-overview/>
