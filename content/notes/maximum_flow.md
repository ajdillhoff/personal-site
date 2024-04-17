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
2.  **Flow conservation:** For all $u &isin; V - \\{s, t\\},
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


## A linear time solution {#a-linear-time-solution}

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Cormen, Thomas H., Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2022. <i>Introduction to Algorithms</i>. 4th ed. MIT Press. <a href="http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/">http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/</a>.</div>
</div>
