+++
title = "Single-Source Shortest Paths"
authors = ["Alex Dillhoff"]
date = 2023-10-21T00:00:00-05:00
tags = ["computer science", "algorithms", "graphs", "shortest path"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Definition](#definition)
- [Bellman-Ford](#bellman-ford)
- [Dijkstra's Algorithm](#dijkstra-s-algorithm)

</div>
<!--endtoc-->

When you hear the term _shortest path_, you may think of the shortest physical distance between your current location and wherever it is you're going. Finding the most optimal route via GPS is one of the most widely used mobile applications. Physical paths are not the only types we may wish to find a shortest path for. Other examples include:

-   **Network Routing**: To improve network performance, it is critical to know the shortest path from one system to another in terms of latency.
-   **Puzzle Solving**: For puzzles such as a Rubik's cube, the vertices could represents states of the cube and edges could correspond to a single move.
-   **Robotics**: Shortest paths in terms of robotics have a lot to do with physical distances, but it could also relate the completing a task efficiently.

These notes will cover classical single-source shortest path algorithms, but first we must formally define the problem.


## Definition {#definition}

Given a weighted, directed graph \\(G = (V, E)\\) with weight function \\(w: E \rightarrow \mathbb{R}\\), a source vertex \\(s \in V\\), and a destination vertex \\(t \in V\\), find the shortest path from \\(s\\) to \\(t\\). The weight of a path is defined as the sum of the weights of its edges:

\\[
w(p) = \sum\_{e \in p} w(e).
\\]

The shortest-path weight between two vertices \\(u\\) and \\(v\\) is given by

\\[
\delta(u, v) = \begin{cases}
\min\_{p \in P(u, v)} w(p) & \text{if } P(u, v) \neq \emptyset \\\\
\infty & \text{otherwise}
\end{cases}
\\]

where \\(P(u, v)\\) is the set of all paths from \\(u\\) to \\(v\\). The shortest-path weight from \\(s\\) to \\(t\\) is given by \\(\delta(s, t)\\).

Shortest-path algorithms rely on an optimal substructure property that is defined by Lemma 22.1 (<a href="#citeproc_bib_item_1">Cormen et al. 2022</a>).

**Lemma 22.1**

> Given a weighted, directed graph \\(G = (V,E)\\) with weight function \\(w: E \rightarrow \mathbb{R}\\), let \\(p = \langle v\_0, v\_1, \dots, v\_k \rangle\\) be a shortest path from vertex \\(v\_0\\) to vertex \\(v\_k\\). For any \\(i\\) and \\(j\\) such that \\(0 \leq i \leq j \leq k\\), let \\(p\_{ij} = \langle v\_i, v\_{i+1}, \dots, v\_j \rangle\\) be the subpath of \\(p\\) from vertex \\(v\_i\\) to vertex \\(v\_j\\). Then, \\(p\_{ij}\\) is a shortest path from \\(v\_i\\) to \\(v\_j\\).

It is also important to note here that a shortest path should contain no cycles. Some shortest-path algorithms require that the edge weights be strictly positive. For those that do not, they may have some mechanism for detecting negative-weight cycles. In any case, a cycle of any kind cannot be included in a shortest path. This is because if a cycle were included, we could simply traverse the cycle as many times as we wanted to reduce the weight of the path. For positive-weight cycles, if a shortest path included a cycle, then surely we could remove the cycle to get a lower weight.

As we build a shortest path, we need to keep track of which vertices lead us from the source to the destination. Some algorithms maintain this by keeping a **predecessor** attribute for each vertex in the path. Solutions such as the Viterbi algorithm keep an array of indices that correspond to the vertices in the path. In any case, we will need to keep track of the vertices in the path as we build it.


### Relaxation {#relaxation}

There is one more important property to define before discussing specific algorithms: **relaxation**. Relaxing an edge \\((u, v)\\) is to test whether going through vertex \\(u\\) improves the shortest path to \\(v\\). If so, we update the shortest-path estimate and predecessor of \\(v\\) to reflect the new shortest path. Relaxation requires that we maintain the shortest-path estimate and processor for each vertex. This is initialized as follows.

```python
def initialize_single_source(G, s):
    for v in G.V:
        v.d = float('inf')
        v.pi = None
    s.d = 0
```

When the values are changed, we say that the vertex has been **relaxed**. Relaxing an edge \\((u, v)\\) is done as follows.

```python
def relax(u, v, w):
    if v.d > u.d + w(u, v):
        v.d = u.d + w(u, v)
        v.pi = u
```


#### Properties {#properties}

Relaxation has the following properties.

-   If the shortest-path estimate of a vertex is not \\(\infty\\), then it is always an upper bound on the weight of a shortest path from the source to that vertex.
-   The shortest-path estimate of a vertex will either stay the same or decrease as the algorithm progresses.
-   Once a vertex's shortest-path estimate is finalized, it will never change.
-   The shortest-path estimate of a vertex is always greater than or equal to the actual shortest-path weight.
-   After \\(i\\) iterations of relaxing on all \\((u, v)\\), if the shortest path to \\(v\\) has \\(i\\) edges, then \\(v.d = \delta(s, v)\\).

    Following _Introduction to Algorithms_, we will first discuss the Bellman-Ford algorithm, which has a higher runtime but works with graphs that have negative edge weights. Then, we will discuss Dijkstra's algorithm, which has a lower runtime but only works with graphs that have non-negative edge weights.


## Bellman-Ford {#bellman-ford}

The Bellman-Ford algorithm is a dynamic programming algorithm that solves the single-source shortest-paths problem in the general case in which edge weights may be negative. If a negative-weight cycle is reachable from the source, then the algorithm will report its existence. Otherwise, it will report the shortest-path weights and predecessors. It works by relaxing edges, decreasing the shortest-path estimate on the weight of a shortest path from \\(s\\) to each vertex \\(v\\) until it reaches the shortest-path weight.

```python
def bellman_ford(G, w, s):
    initialize_single_source(G, s)
    for i in range(1, len(G.V)):
        for (u, v) in G.E:
            relax(u, v, w)
    for (u, v) in G.E:
        if v.d > u.d + w(u, v):
            return False
    return True
```


### Example {#example}

In the figure below, graph (a) shows the original graph before iterating over the edges. Graphs (b)-(e) show the result of looping over both edges originating from \\(s\\). Depending on the implementation, the first iteration of the vertices would result directly in graph (c). You can find a Python implementation of this example [here](<https://github.com/ajdillhoff/python-examples/blob/main/data_structures/graphs/bellman_ford_algorithm.ipynb>).

{{< figure src="/ox-hugo/2023-10-24_21-05-33_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Step-by-step execution of Bellman-Ford on a graph with negative-weight edges (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}


### Analysis {#analysis}

Using an adjacency list representation, the runtime of Bellman-Ford is \\(O(V^2 + VE)\\). The initialization takes \\(\Theta(V)\\). Each of the \\(|V| - 1\\) iterations over the edges takes \\(\Theta(V + E)\\), and the final check for negative-weight cycles takes \\(\Theta(V + E)\\). If the number of edges and vertices is such that the number of vertices are a lower bound on the edges, then the runtime is \\(O(VE)\\).


## Dijkstra's Algorithm {#dijkstra-s-algorithm}

Dijkstra's algorithm also solves the single-source shortest path problem on a weighted, directed graph \\(G = (V,E)\\) but requires nonnegative weights on all edges. It works in a breadth-first manner. A minimum priority queue is utilized to keep track of the vertices that have not been visited based on their current minimum shortest-path estimate. The algorithm works by relaxing edges, decreasing the shortest-path estimate on the weight of a shortest path from \\(s\\) to each vertex \\(v\\) until it reaches the shortest-path weight.

```python
def dijkstra(G, w, s):
    initialize_single_source(G, s)
    S = []
    Q = G.V
    while Q:
        u = extract_min(Q)
        S.append(u)
        for v in G.adj[u]:
            prev_d = v.d
            relax(u, v, w)
            if v.d < prev_d:
                decrease_key(Q, v)
```


### Example {#example}

A Python example of the figure below is available [here](<https://github.com/ajdillhoff/python-examples/blob/main/data_structures/graphs/dijkstras_algorithm.ipynb>).

{{< figure src="/ox-hugo/2023-10-25_08-21-04_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>A step-by-step execution of Dijkstra's algorithm on a graph with non-negative edge weights (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}


### Analysis {#analysis}

See Chapter 22 of _Introduction to Algorithms_ for a detailed analysis of Dijkstra's algorithm. Inserting the nodes and then extracting them from the queue yields \\(O(V \log V)\\). After extracting a node, its edges are iterated with a possible update to the queue. This takes \\(O(E \log V)\\). The total runtime is \\(O((V + E) \log V)\\).

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Cormen, Thomas H., Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2022. <i>Introduction to Algorithms</i>. 4th ed. MIT Press. <a href="http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/">http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/</a>.</div>
</div>
