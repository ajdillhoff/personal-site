+++
title = "Introduction to Graph Theory"
authors = ["Alex Dillhoff"]
date = 2023-10-17T00:00:00-05:00
tags = ["computer science", "data structures"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [What are Graphs?](#what-are-graphs)
- [Graph Traversal Algorithms](#graph-traversal-algorithms)

</div>
<!--endtoc-->



## What are Graphs? {#what-are-graphs}

A **graph** is a data structure that is used to represent pairwise relationships between objects. Graphs are used in many applications, such as social networks, maps, and routing algorithms. These notes accompany the series of lectures on graphs for my _Foundations of Computing_ course at the University of Texas - Arlington.


### Definitions {#definitions}

-   A **directed graph** \\(G\\) is represented as a pair \\((V, E)\\) of a set of vertices \\(V\\) and edges \\(E\\). Edges are represented as ordered pairs.
-   An **undirected graph** \\(G\\) is represented as a pair \\((V, E)\\) of a set of vertices \\(V\\) and edges \\(E\\). The edges are represented as unordered pairs, as it does not matter which direction the edge is going.
-   Let \\((u, v)\\) be an edge in a graph \\(G\\). If \\(G\\) is a directed graph, then the edge is **incident from** \\(u\\) and is **incident to** \\(v\\). In this case, \\(v\\) is also **adjacent** to \\(u\\). If \\(G\\) is an undirected graph, then the edge is **incident on** \\(u\\) and \\(v\\). For undirected graphs, the **adjacency** relation is symmetric.
-   The **degree** is a graph is the number of edges incident on a vertex. For directed graphs, the **in-degree** is the number of edges incident to a vertex, and the **out-degree** is the number of edges incident from a vertex.
-   A **path** from a vertex \\(u\\) to another vertex \\(v\\) is a sequence of edges that starts at \\(u\\) and ends at \\(v\\). This definition can include duplicates. A **simple path** is a path that does not repeat any vertices. A **cycle** is a path that starts and ends at the same vertex. If a path exists from \\(u\\) to \\(v\\), then \\(u\\) is **reachable** from \\(v\\).
-   A **connected graph** is a graph where there is a path between every pair of vertices. A **strongly connected graph** is a directed graph where there is a path between every pair of vertices. The **connected components** of a graph are the subgraphs in which each pair of nodes is connected by a path. In image processing, connected-component labeling is used to find regions of connected pixels in a binary image.
-   Let \\(G = (V, E)\\) and \\(G' = (V', E')\\). \\(G\\) and \\(G'\\) are **isomorphic** if there is a bijection between their vertices such that \\((u, v) \in E\\) if and only if \\((f(u), f(v)) \in E'\\).
-   A **complete graph** is an undirected graph in which every pair of vertices is adjacent. A **bipartite graph** is an undirected graph in which the vertices can be partitioned into two sets such that every edge connects a vertex in one set to a vertex in the other set.
-   A **multi-graph** is a graph that allows multiple edges between the same pair of vertices. These are commonly in social network analysis, where multiple edges between two people can represent different types of relationships.

    TODO: Add figures demonstrating the above definitions


### Representations {#representations}

Graphs can be represented in many different ways. The most common representations are adjacency lists and adjacency matrices. Adjacency lists are more space-efficient for sparse graphs, while adjacency matrices are more space-efficient for dense graphs. Adjacency lists are also more efficient for finding the neighbors of a vertex, while adjacency matrices are more efficient for checking if an edge exists between two vertices.


#### Example: Adjacency Matrix and Reachability {#example-adjacency-matrix-and-reachability}

Consider the graph in the figure below.

{{< figure src="/ox-hugo/2023-10-17_21-08-29_directed_graph.png" caption="<span class=\"figure-number\">Figure 1: </span>A directed graph" >}}

The adjacency matrix for this graph is:

\begin{bmatrix}
0 & 0 & 0 & 1 & 0 & 0 & 0\\\\
0 & 0 & 0 & 1 & 0 & 0 & 0\\\\
0 & 0 & 0 & 0 & 1 & 0 & 0\\\\
0 & 0 & 0 & 0 & 0 & 1 & 1\\\\
0 & 0 & 0 & 0 & 0 & 0 & 1\\\\
0 & 0 & 0 & 0 & 0 & 0 & 1\\\\
0 & 0 & 0 & 0 & 0 & 0 & 0\\\\
\end{bmatrix}

The rows and columns represent the vertices in the graph. The value at row \\(i\\) and column \\(j\\) is 1 if there is an edge from vertex \\(i\\) to vertex \\(j\\). Otherwise, the value is 0. Let \\(A\\) be the adjacency matrix for a graph \\(G\\). The matrix \\(A^k\\) represents the number of paths of length \\(k\\) between each pair of vertices. For example, \\(A^2\\) for the above graph is:

\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 1 & 1\\\\
0 & 0 & 0 & 0 & 0 & 1 & 1\\\\
0 & 0 & 0 & 0 & 0 & 0 & 1\\\\
0 & 0 & 0 & 0 & 0 & 0 & 1\\\\
0 & 0 & 0 & 0 & 0 & 0 & 0\\\\
0 & 0 & 0 & 0 & 0 & 0 & 0\\\\
0 & 0 & 0 & 0 & 0 & 0 & 0\\\\
\end{bmatrix}

The value at row \\(i\\) and column \\(j\\) is the number of paths of length 2 from vertex \\(i\\) to vertex \\(j\\). For example, is a path from vertex 0 to vertex 6 via 0 -&gt; 3 -&gt; 6.


## Graph Traversal Algorithms {#graph-traversal-algorithms}


### Breadth First Search {#breadth-first-search}

We previously studied breadth-first search in the context of binary search trees. The algorithm is the same when applied on general graphs, but our perspective is slightly different now. The function studied before did not use node coloring. Let's investigate the algorithm given by Cormen et al. in _Introduction to Algorithms_.

The algorithm adds a color to each node to keep track of its state. The colors are:

-   WHITE: The node has not been discovered yet.
-   GRAY: The node has been discovered, but not all of its neighbors have been discovered.
-   BLACK: The node has been discovered, and all of its neighbors have been discovered.

First, every vertex is painted white and the distance is set to \\(\infty\\). The first node `s` is immediately set to have 0 distance. The queue then starts with `s`. While there are any grey vertices, dequeue the next available node and add its adjacent vertices to the queue. The distance of each adjacent vertex is set to the distance of the current vertex plus one. Once all of its neighbors have been discovered, the current vertex is painted black.

The algorithm and an example run from Cormen et al. are shown below.

```python
def bfs(G, s):
    for u in G.V:
        u.color = WHITE
        u.d = inf
        u.pi = None
    s.color = GRAY
    s.d = 0
    s.pi = None
    Q = Queue()
    Q.enqueue(s)
    while not Q.empty():
        u = Q.dequeue()
        for v in G.adj[u]:
            if v.color == WHITE:
                v.color = GRAY
                v.d = u.d + 1
                v.pi = u
                Q.enqueue(v)
        u.color = BLACK
```

{{< figure src="/ox-hugo/2023-10-17_20-11-28_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Breadth First Search from Cormen et al." >}}


#### Analysis {#analysis}

The running time of BFS is \\(O(V + E)\\), where \\(V\\) is the number of vertices and \\(E\\) is the number of edges. Each vertex is queued and dequeued once, so the queue operations take \\(O(V)\\) time. Each edge is examined once, so the edge operations take \\(O(E)\\) time. The total running time is \\(O(V + E)\\).


### Depth First Search {#depth-first-search}

Like the BFS algorithm presented in _Introduction to Algorithms_ by Cormen et al., the DFS algorithm also uses colors to keep track of the state of each node. The colors are similar to the BFS algorithm, but the meaning is slightly different:

-   WHITE: The node has not been discovered yet.
-   GRAY: The node has been visited for the first time.
-   BLACK: The adjacency list of the node has been examined completely.

First, all vertices are colored white. The time is set to 0. The function `dfs_visit` is called on each vertex. The function is defined as follows:

```python
def dfs_visit(G, u):
    time += 1
    u.d = time
    u.color = GRAY
    for v in G.adj[u]:
        if v.color == WHITE:
            v.pi = u
            dfs_visit(G, v)
    u.color = BLACK
    time += 1
    u.f = time
```

When a node is discovered via `dfs_visit`, the time is recorded and the color is changed to gray. The start and finish times are useful in understanding the structure of the graph. After all of the node's neighbors have been discovered, the color is changed to black and the finish time is recorded. That is, the depth from the current node must be fully explored before it is considered finished. The figure below shows the progress of DFS on a directed graph.

{{< figure src="/ox-hugo/2023-10-17_20-25-40_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Depth First Search from Cormen et al." >}}
