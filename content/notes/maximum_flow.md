+++
title = "Maximum Flow"
authors = ["Alex Dillhoff"]
date = 2024-04-12T18:51:00-05:00
tags = ["algorithms", "computer science"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Objective Questions](#objective-questions)
- [Maximum Flow](#maximum-flow)
- [A polynomial time solution](#a-polynomial-time-solution)
- [A linear time solution](#a-linear-time-solution)

</div>
<!--endtoc-->

A flow network is a directed graph in which the edges begin at a node that produces the flow and the adjacent nodes are the ones that receive it. _Flow_ in this context could take on many meanings, such as the amount of water that can flow through a pipe, the amount of data that can be sent through a network, or the amount of traffic that can be sent through a road network. The goal of a flow network is to maximize the flow from the source to the sink.

The problem may have intermediate constraints. For example, a network graph may have a node with limited bandwidth, so the flow through that node must be less than or equal to the bandwidth. These notes review the formal definition of the problem followed by a solution using the Ford-Fulkerson algorithm as well as one related to bipartite matching.


## Objective Questions {#objective-questions}

-   [ ] What is the **maximum flow** problem?
-   [ ] How can we solve it?


## Maximum Flow {#maximum-flow}

A **flow network** \\(G = (V, E)\\) is a directed graph in which each edge \\((u, v) \in E\\) has a nonnegative **capacity** \\(c(u, v) \geq 0\\). The graph does not contain reverse edges between two vertices. If an edge does not exist in the set, then its capacity is 0. Each graph has a **source** and a **sink**, which will be the main edges of note when analyzing the graph. The goal is to maximize the flow going from the source to the sink. This implies that the source has no incoming edges.

{{< figure src="/ox-hugo/2024-04-13_19-02-08_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>A flow network. Each edge depicts \\(f(u,v)/c(u,v)\\), the flow and capacity (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}

A **flow** in a graph \\(G\\) satisfies two properties:

1.  **Capacity constraint:** For all \\(u, v \in V\\),
    \\[
       0 \leq f(u,v) \leq c(u,v).
       \\]
2.  **Flow conservation:** For all \\(u \in V - \\{s, t\\}\\),
    \\[
       \sum\_{v \in V} f(v, u) = \sum\_{v \in V} f(u, v).
       \\]

There are usually many different possibly paths of flow in a flow network. The **maximum flow problem** asks: what is the path that yields the maximum flow?


### Antiparallel Edges {#antiparallel-edges}

The restriction that no two nodes may have more than one edge seems to be unrealistic. For example, modeling the flow of a network graph with this restriction means that network traffic can only move in one direction between two datacenters. If there were two edges between adjacent nodes \\(v\_1\\) and \\(v\_2\\) such that \\((v\_1, v\_2) \in E\\) and \\((v\_1, v\_2) \in E\\), we would call these edges **antiparallel**. In such cases, the graph is modified with a new node \\(v'\\) such that \\((v\_1, v') \in E\\) and \\((v', v\_2) \in E\\). An example is shown below.

{{< figure src="/ox-hugo/2024-04-13_19-23-43_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Addressing antiparallel edges in a flow network (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}


### Multiple Sources and Sinks {#multiple-sources-and-sinks}

Another restriction that is unrealistic in many real-world scenarios is that maximum flow graphs can only have a single source and sink. It is easy to imagine a scenario where multiple sources and sinks within a network. Again, the graph can be modified to accommodate this scenario by defining a **supersource** and **supersink** whose outgoing and incoming flows are infinite.

{{< figure src="/ox-hugo/2024-04-13_19-27-09_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Assing a supersource and supersink to a graph with multiple sources and sinks (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}


## A polynomial time solution {#a-polynomial-time-solution}

Ford-Fulkerson relies on three foundational concepts:

1.  Residual networks
2.  Augmenting paths
3.  Cuts

The solution presented by Ford and Fulkerson is not a single algorithm but rather a set of general instructions.

1.  Initialize \\(f(u, v) = 0\\) for all \\(u,v \in V\\), giving an initial flow of 0.
2.  Increase the flow by finding an **augmenting path** in a **residual network**.
3.  The edges of the augmented path indicate where to increase the flow.

This is repeated until the residual network has no more augmenting paths.


### Residual Networks {#residual-networks}

Consider a pair of vertices \\(u, v \in V\\), the residual capacity \\(c\_f(u, v)\\) is amount of additional flow that can be pushed from \\(u\\) to \\(v\\).

\\[
c\_f(u, v)= \begin{cases}
c(u,v) - f(u, v)\quad \text{if } (u,v) \in E,\\\\
f(v, u)\quad \text{if } (v, u) \in E,\\\\
0\quad \text{otherwise (i.e., } (u, v), (v, u) \notin E).
   \end{cases}
\\]

The **residual network** is \\(G\_f = (V, E\_f)\\), where

\\[
E\_f= \\{(u, v) \in V \times V : c\_f(u, v) > 0\\}.
\\]

The edges of \\(G\_f\\) represent those edges in \\(G\\) with the capacity to change the flow. There is also no requirement for all edges in \\(G\\) to be present in \\(G\_f\\). As the algorithm works out the solution, we are only considered with edges that permit more flow.

An edge \\((u, v) \in E\\) means that the reverse edge \\((v, u) \notin E\\). However, the residual network can have edges that are not in \\(G\\). These are used to represent paths in which flow is sent in the reverse direction. This can happen if reducing flow from one edge results in a net increase across some other.

In \\(G\_f\\), the reverse edges \\((v, u)\\) represent the flow on \\((u, v) \in G\\) that could be sent back.

{{< figure src="/ox-hugo/2024-04-19_15-29-03_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>A flow network and its residual network (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}


#### Augmentation Function {#augmentation-function}

Given flows \\(f\\) in \\(G\\) and \\(f'\\) in \\(G\_f\\), define the **augmentation** of \\(f\\) by \\(f'\\), as a function \\(V \times V \rightarrow \mathbb{R}\\):

\\[
(f \uparrow f')(u, v) = \begin{cases}
        f(u, v) + f'(u, v) - f'(u, v)\quad \text{it } (u, v) \in E,\\\\
        0 \quad \text{otherwise}
   \end{cases}
\\]

for all \\((u, v) \in V\\).

This augmentation function represents an increase of flow on \\((u, v)\\) by \\(f'(u, v)\\) with a decrease by \\(f'(v, u)\\) since pushing flow on the reverse edge in \\(G\_f\\) represents a decrease in \\(G\\). This is known as **cancellation**.

**Lemma**

Given a flow network \\(G\\), a flow \\(f\\) in \\(G\\), and the residual network \\(G\_f\\), let \\(f'\\) be a flow in \\(G\_f\\). Then \\((f \uparrow f')\\) is a flow in \\(G\\) with value \\(|f \uparrow f'| = |f| + |f'|\\).


## A linear time solution {#a-linear-time-solution}

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Cormen, Thomas H., Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2022. <i>Introduction to Algorithms</i>. 4th ed. MIT Press. <a href="http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/">http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/</a>.</div>
</div>
