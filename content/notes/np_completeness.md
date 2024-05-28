+++
title = "NP-Completeness"
authors = ["Alex Dillhoff"]
date = 2024-04-25T11:03:00-05:00
tags = ["algorithms", "computer science"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Formal Languages](#formal-languages)
- [Reductions](#reductions)
- [Clique Problem](#clique-problem)
- [Vertex Cover Problem](#vertex-cover-problem)

</div>
<!--endtoc-->



## Introduction {#introduction}

Most of the algorithms discussed in a typical algorithms course run in polynomial time. This focus is reasonable since algorithms that run worse than polynomial time have little practical use. To simplify this notion: a problem for which a polynomial-time algorithm exists is "easy" and a problem for which no polynomial-time algorithm exists is "hard". Knowing how to determine whether a problem is easy or hard is extremely useful. If one can identify a hard problem, then an approximate solution may be the best that can be achieved.

One of the most fundamental problems in computer science is the classification of problems into these two categories. These notes provide an introduction to this classification.


### P, NP, and NP-Complete {#p-np-and-np-complete}

There are three classes of algorithms:

-   Polynomial-time
-   NP (nondeterministic polynomial time)
-   NP-complete

Problems in P are those solvable in polynomial time. This means _any_ constant \\(k\\) such that the running time is \\(O(n^k)\\).

The class NP is a superset of P. These are problems that can be **verified** in polynomial time. This means that if someone gives you a solution to the problem, you can verify that it is correct in polynomial time. This is different from solving the problem in polynomial time. Problems in NP can be solved in **nondeterministic** polynomial time. However, such a model of computation does not exist in the real world.

NP-Complete problems are problems in NP that are as _hard_ as any other problem in NP. This means that if you can solve an NP-Complete problem in polynomial time, you can solve any problem in NP in polynomial time. This is why NP-Complete problems are so important.


### Verifying a Solution {#verifying-a-solution}

As long as we can come up with a verification algorithm for a problem in polynomial time, we can say that the problem is in \\(NP\\). This is true even if we later find a polynomial-time algorithm for the problem.


### Proving that a problem is NP-Complete {#proving-that-a-problem-is-np-complete}

Proving that a problem belongs to either NP or NPC is difficult the first time you do it. Luckily, now that problems have been proven to be NP-Complete, you can use these problems to prove that other problems are NP-Complete. First, let's introduce one more class: NP-Hard. Informally, a problem \\(X\\) is NP-Hard if it is at least as hard as any problem in NP. If we can reduce every problem \\(Y \in NP\\) to \\(X\\) in polynomial time, then \\(X\\) is NP-Hard. If \\(X\\) is also in NP, then \\(X\\) is NP-Complete.


### Optimization versus decision problems {#optimization-versus-decision-problems}

Many problems are framed as optimization problems. Given some criteria, the goal is to find the best solution according to that criteria. For the shortest path problem, the algorithm finds a path between two vertices in the fewest number of edges. One can intuit that this is a slightly harder problem than that of determining if a path exists using only \\(k\\) edges. This latter problem is a decision problem.

The reason this is worth talking about is that decision problems are often easier to come up with than optimization problems. If one can provide that a decision problem is hard, then its optimization problem is also hard.


### Reducing one problem to another {#reducing-one-problem-to-another}

A common strategy for relating two problems is to reduce one to the other. For example, if problem \\(B\\) runs in polynomial time, and we can reduce problem \\(A\\) to problem \\(B\\) in polynomial time, then problem \\(A\\) is also in P. This is because we can solve \\(A\\) by reducing it to \\(B\\) and then solving \\(B\\) in polynomial time.


## Formal Languages {#formal-languages}


## Reductions {#reductions}

The main idea behind reductions is to first show that a problem is NP-Complete. This first proof was done by Cook in 1971. With a problem proven to be NP-Complete, one can then show that other problems are NP-Complete by reducing them to the first problem. This method is far simpler than the original proof and provides a convenient process for proving that a problem is NP-Complete. The process is based on the following lemma.

**Lemma 34.8**

If \\(L\\) is a language such that \\(L' \leq\_p L\\) for some \\(L' \in \text{NPC}\\), then \\(L\\) is NP-hard. If, in addition, we have \\(L \in \text{NP}\\), then \\(L\\) is NP-Complete.


### Circuit Satisfiability {#circuit-satisfiability}

-   What is this problem?
-   Why is it important?
-   How is it related to NP-Completeness?

Given a boolean combinatorial circuit, is it satisfiable. That is, is there an assignment of values to the inputs that makes the output true?

This problem is in the class NP. To prove this, we only need to show that given an assignment of values to the inputs, we can verify that the output is true in polynomial time. For a given circuit, we can verify that the output is true by following the circuit from the inputs to the outputs.

Proving that it is NP-Complete is much more difficult, so a brief overview is provided here. Let \\(A\\) be any problem in \\(NP\\). Since any problem in \\(NP\\) has a polynomial-time verification algorithm, we can construct a boolean circuit that simulates this algorithm. This circuit will have a single output that is true if and only if the input is a valid solution to the problem. This circuit is satisfiable if and only if the input is a valid solution to the problem. Therefore, the circuit satisfiability problem is NP-Complete.


#### Example: Exercise 34.3-1 {#example-exercise-34-dot-3-1}

Verify that the given circuit is unsatisfiable.

{{< figure src="/ox-hugo/2024-04-27_14-31-53_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Figure 34.8 from (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}

For reference, here are definitions for each of the gates listed in the figure.

{{< figure src="/ox-hugo/2024-04-27_14-35-18_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Definitions for the gates in the circuit from Figure 34.7 (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}

**How can we prove this circuit is unsatisfiable?**

The easiest way to do this is to code it up and brute force it.


### Formula Satisfiability {#formula-satisfiability}

-   Satisfiability (SAT) is NP-Hard.
-   Can show that CIRCUIT-SAT \\(\leq\_p\\) SAT.
-   Then show that SAT \\(\leq\_p\\) 3SAT.

An instance of the **formula satisfiability (SAT)** problem is a boolean formula \\(\phi\\) with

-   \\(n\\) variables \\(x\_1, x\_2, \ldots, x\_n\\)
-   \\(m\\) clauses \\(C\_1, C\_2, \ldots, C\_m\\)

For example, the formula

\\[
\phi = (x\_1 \lor x\_2) \land (\lnot x\_1 \lor x\_3) \land (x\_2 \lor x\_3)
\\]

has the satisfying assignment \\(x\_1 = 1, x\_2 = 0, x\_3 = 1\\).


#### SAT belongs to NP {#sat-belongs-to-np}

Showing that SAT is in NP is straightforward. Given a boolean formula \\(\phi\\) and an assignment of values to the variables, one can verify that the formula is satisfied in polynomial time. This is enough to show that SAT is in NP.


#### CIRCUIT-SAT \\(\leq\_p\\) SAT {#circuit-sat-leq-p-sat}

If we can reduce an instance of CIRCUIT-SAT, which is known to be NP-Complete, to SAT, then SAT is also NP-Complete. This proof is by contradiction: assume that SAT is not NP-Complete. Then, CIRCUIT-SAT is not NP-Complete. But we know that CIRCUIT-SAT is NP-Complete, so this is a contradiction.

The reduction starts by introducing a variable for each wire and a clause for each gate, as seen below.

{{< figure src="/ox-hugo/2024-04-27_15-34-18_screenshot.png" >}}

The reduction algorithm produces a formula for each gate in terms of an "if and only if" statement.

\begin{align\*}
\phi = x\_{10} &\land (x\_4 \leftrightarrow \lnot x\_3)\\\\
&\land (x\_5 \leftrightarrow (x\_1 \lor x\_2))\\\\
&\land (x\_6 \leftrightarrow \lnot x\_4)\\\\
&\land (x\_7 \leftrightarrow (x\_1 \land x\_2 \land x\_4))\\\\
&\land (x\_8 \leftrightarrow (x\_5 \lor x\_6))\\\\
&\land (x\_9 \leftrightarrow (x\_6 \lor x\_7))\\\\
&\land (x\_{10} \leftrightarrow (x\_6 \land x\_8 \land x\_9))\\\\
\end{align\*}

A simpler explanation for this reduction is that a circuit can be represented as a boolean formula. This formula can be solved by the SAT algorithm.


### 3SAT {#3sat}

The 3SAT problem is a special case of SAT where each clause has exactly three literals. This problem is also NP-Complete. Many problems can be reduced to 3SAT, which is why it is so important.


#### Definition {#definition}

An instance of 3SAT is a boolean formula \\(\phi\\) with

-   \\(n\\) **literals** \\(x\_1, x\_2, \ldots, x\_n\\)
-   \\(m\\) **clauses** \\(C\_1, C\_2, \ldots, C\_m\\)

Each clause has exactly three literals and is in **conjunctive normal form** (CNF), which means it is expressed as an AND of clauses. For example, the formula

\\[
\phi = (x\_1 \lor x\_2 \lor x\_3) \land (\lnot x\_1 \lor x\_2 \lor x\_3) \land (x\_1 \lor \lnot x\_2 \lor x\_3)
\\]

is a 3SAT formula.


#### 3SAT is NP-Complete {#3sat-is-np-complete}

The 3SAT problem is NP-Complete. This can be shown by reducing SAT to 3SAT. A thorough proof is provided in the textbook (<a href="#citeproc_bib_item_1">Cormen et al. 2022</a>).


## Clique Problem {#clique-problem}

A clique is a **complete subgraph** of an undirected graph \\(G\\). That is, a clique is a set of vertices such that every pair of vertices is connected by an edge. The **clique problem** is to find the largest clique in a graph.

\\[
\text{CLIQUE} = \\{ \langle G, k \rangle \mid G \text{ has a clique of size } k \\}
\\]


### Clique is in NP {#clique-is-in-np}

Let's say you have access to a clique of size \\(k\\). You can verify that this is a clique in polynomial time by checking that every pair of vertices is connected by an edge. That is, for each pair \\(u, v \in V'\\), the edge \\((u, v)\\) is in \\(E\\), where \\(V'\\) is the set of vertices in the clique.

Thus, we have a polynomial-time verification algorithm for the clique problem, so it is in NP.


### Clique is NP-Complete {#clique-is-np-complete}

Knowing that 3SAT is NP-Complete, we can reduce 3SAT to the clique problem. The reduction may not be intuitive as a boolean formula seems to have no relation to a graph.

Let \\(\phi = C\_1 \land C\_2 \land \ldots \land C\_m\\) be a 3SAT formula with \\(n\\) variables. We construct a graph \\(G\\) with \\(3n\\) vertices. Each \\(C\_r\\) has three literals, \\(l\_1^r, l\_2^r, l\_3^r\\).

To construct the graph, we create a triplet of vertices \\(v\_1^r, v\_2^r, v\_3^r\\) for each clause \\(C\_r\\) such that there is no edge connecting any two vertices in the same triplet. There is an edge \\((v\_i^r, v\_j^s) \in E\\) if

1.  \\(v\_i^r\\) and \\(v\_j^s\\) are in different triplets
2.  \\(l\_i^r\\) and \\(l\_j^s\\) are not negations of each other

One such formula that we can convert is

\\[
\phi = (x\_1 \lor \lnot x\_2 \lor \lnot x\_3) \land (\lnot x\_1 \lor x\_2 \lor x\_3) \land (x\_1 \lor x\_2 \lor x\_3).
\\]

The resulting graph is shown below.

{{< figure src="/ox-hugo/2024-05-01_09-44-28_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Graph constructed from the 3SAT formula \\(\phi\\) (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}

If a satisfying assignment exists for \\(\phi\\), then each \\(C\_r\\) has at least one literal that is true. Consider the corresponding vertices in the graph for a satisfying assignment. Since there is at least one true literal in each clause, there is at least one edge between the corresponding vertices. Thus, a reduction from 3SAT to the clique problem is possible.

**How does this show that Clique is NP-Complete?**

It is true that this example shows a very specialized graph. However, this is enough to show that the problem is NPC. If there were a polynomial time solution for Clique on a general graph \\(G\\), then surely it would work for a specialized graph as well.

If it could solve this one, then the corresponding 3SAT formula would be solvable as well. This is a contradiction, so the Clique problem is NP-Complete.


## Vertex Cover Problem {#vertex-cover-problem}

{{< figure src="/ox-hugo/2024-05-01_09-49-06_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Vertex cover examples (Wikipedia)." >}}

The **vertex cover problem** is to find the smallest set of vertices such that every edge in the graph is incident to at least one vertex in the set. More formally, a vertex cover of a graph \\(G\\) is a set \\(V' \subseteq V\\) such that for every edge \\((u, v) \in E\\), either \\(u \in V'\\) or \\(v \in V'\\).


### Vertex Cover is in NP {#vertex-cover-is-in-np}

Given a set of vertices \\(V'\\), one can verify that it is a vertex cover in polynomial time by checking that every edge is incident to at least one vertex in the set. This is a polynomial-time verification algorithm, so the vertex cover problem is in NP.


### Vertex Cover is NP-Complete {#vertex-cover-is-np-complete}

We can show that the vertex cover problem is NP-Complete by reducing it to an instance of the clique problem. For this, we need to introduce the definition of a graph complement. Given a graph \\(G = (V, E)\\), the **complement** of \\(G\\) is the graph \\(\overline{G} = (V, E')\\) where \\(E' = \\{ (u, v) \mid u, v \in V \text{ and } (u, v) \notin E \\}\\). Basically, \\(\overline{G}\\) has all the edges that \\(G\\) does not have.

Let \\(G\\) contain a clique \\(V' \subseteq V\\), where \\(|V'| = k\\). Then \\(V - V'\\) is a vertex cover of \\(\overline{G}\\). If \\((u, v) \in \overline{E}\\), but is not in \\(E\\), then at least one of \\(u\\) or \\(v\\) is not in \\(V'\\).

{{< figure src="/ox-hugo/2024-05-01_09-54-50_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>Graph \\(G\\) and its complement \\(\overline{G}\\) (<a href=\"#citeproc_bib_item_1\">Cormen et al. 2022</a>)." >}}

In the example above, any edge in \\(\overline{G}\\) has at least one vertex that is not in \\(G\\). On the same edge, at least one is in \\(V - V'\\), implying that \\((u, v)\\) is covered by \\(V - V'\\). Thus, a reduction from the clique problem to the vertex cover problem is possible.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Cormen, Thomas H., Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2022. <i>Introduction to Algorithms</i>. 4th ed. MIT Press. <a href="http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/">http://mitpress.mit.edu/9780262046305/introduction-to-algorithms/</a>.</div>
</div>
