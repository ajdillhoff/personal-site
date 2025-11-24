+++
title = "Bipartite Matching"
authors = ["Alex Dillhoff"]
date = 2025-11-23T14:56:00-06:00
tags = ["algorithms", "computer science"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Bipartite Graphs](#bipartite-graphs)
- [Finding Matchings](#finding-matchings)

</div>
<!--endtoc-->



## Bipartite Graphs {#bipartite-graphs}

A ****bipartite graph**** is a graph whose vertices can be divided into two disjoint and independent sets \\(U\\) and \\(V\\) such that every edge connects a vertex in \\(U\\) to one in \\(V\\). There are no edges between vertices within the same set. The ****matching**** problem is one you are likely familiar with: given a set of words and a set of definitions, can you pair each word with its correct definition? When it comes to graphs, the matching problem involves finding a set of edges such that no two edges share a common vertex.

Formally, given an undirected graph \\(G = (V, E)\\), a ****matching**** \\(M\\) is a subset of edges \\(M \subseteq E\\) such that no two edges in \\(M\\) share a common vertex. A matching is said to be ****maximum**** if it contains the largest possible number of edges.

{{< figure src="/ox-hugo/2025-11-23_17-52-09_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>A matching between \\(L\\) and \\(R\\) in a bipartite graph (left) and a maximum matching (right) (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}


## Finding Matchings {#finding-matchings}

A bipartite graph may have many possible candidate edges between nodes. The algorithms discussed herein take an incremental approach that build up a matching one edge at a time. The two primary algorithms for finding maximum matchings in bipartite graphs are the Hopcroft-Karp algorithm and the Hungarian algorithm.

These algorithms rely on the concept of alternating and augmenting paths. An ****alternating path**** is a path that alternates between edges that are in the matching and edges that are not. An ****augmenting path**** is an alternating path that starts and ends with unmatched vertices.

In the figure below, the alternating path is \\(\\{(l\_5, r\_5), (l\_5, r\_7), (l\_6, r\_7), (l\_7, r\_5), (l\_7, r\_8)\\}\\). The first and last edges are not in the matching, making this an augmenting path.

{{< figure src="/ox-hugo/2025-11-23_18-24-12_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>An augmenting path (highlighted in orange)." >}}

Looking at this figure more closely, you can observe that there are 3 edges in the unmatched set compared to 2 in the matched set. By swapping the matched and unmatched edges along this path, we can increase the size of the matching by 1. This process is known as ****augmenting the matching****. However, you might think this an exception rather than the rule. We can formalize this observation, but we first need a helpful definition.

A **symmetric difference** of two sets \\(A\\) and \\(B\\), denoted by \\(A \oplus B\\), is the set of elements that are in either of the sets but not in their intersection. In other words, it contains elements that are in \\(A\\) or \\(B\\) but not in both.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Cormen, Thomas H., Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2022. <i>Introduction to Algorithms</i>. 4th ed. MIT Press.</div>
</div>
