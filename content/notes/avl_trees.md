+++
title = "AVL Trees"
authors = ["Alex Dillhoff"]
date = 2024-10-31T13:30:00-05:00
tags = ["computer science", "data structures"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Definition](#definition)
- [Operations](#operations)
- [Rebalancing](#rebalancing)

</div>
<!--endtoc-->



## Definition {#definition}

An AVL tree is a binary search tree that is self-balancing based on the height of the tree. It manages this by adding a balance factor property to each node. Given a node \\(X\\), the balance factor is defined as:

\\[
\text{BF}(X) = \text{Height}(\text{Left}(X)) - \text{Height}(\text{Right}(X))
\\]

An binary tree is \textbf{left-heavy} when \\(\text{BF}(X) < 0\\) and \textbf{right-heavy} when \\(\text{BF}(X) > 0\\). An AVL tree is a binary search tree that is balanced if for every node \\(X\\), \\(|\text{BF}(X)| \leq 1\\).


## Operations {#operations}


### Search and Traversal {#search-and-traversal}

Searching in an AVL tree is the same as searching with [Binary Search Trees]({{< relref "binary_search_trees.md" >}}).


### Insertion {#insertion}

Insertion largely follows the same process as with binary search trees. The key difference is based on the requirement to maintain the balance property. After inserting a new node, the balance factor of each node in the path from the root to the new node must be checked. If the balance factor of any node is greater than 1 or less than -1, the tree must be rebalanced.


### Deletion {#deletion}

After a node is removed in the style of binary search trees, rebalancing may be necessary. At the very least, the balance factor of each node in the path from the root to the removed node must be updated.


## Rebalancing {#rebalancing}

If a node is added to a perfectly balanced AVL tree, where \\(BF(X) = 0\\) for all nodes \\(X\\), the tree will remain balanced. Rebalancing then is only necessary when a node is added to a tree with a balance factor of 1 or -1. There are four possible cases for rebalancing an AVL tree based on the balance factor of \\(X\\):

1.  **Left-Left (LL)** - \\(X\\) is left-heavy and its left child is higher than its right child.
2.  **Right-Right (RR)** - \\(X\\) is right-heavy and its right child is higher than its left child.
3.  **Left-Right (LR)** - \\(X\\) is left-heavy and its right child is higher than its left child.
4.  **Right-Left (RL)** - \\(X\\) is right-heavy and its left child is higher than its right child.


### Left-Left and Right-Right {#left-left-and-right-right}

Consider a new node added to an AVL tree that causes an RR violation. This is shown in the figure below. The top half shows the tree during the violation. This was caused by either deleting a node from \\(t\_1\\) or adding a node to \\(t\_4\\). In this case a left rotation is performed to correct the violation.

There is still the matter of updating the balance factors of the \\(X\\) and \\(Z\\). If \\(t\_{23}\\) was previously shorter than \\(t\_4\\), the balance factor of \\(X\\) and \\(Z\\) after the rotation will be 0. If \\(t\_{23}\\) was the same height as \\(t\_4\\), the balance factor of \\(X\\) will be +1 and \\(Z\\) will be -1.

{{< figure src="/ox-hugo/2024-11-03_15-20-45_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>An AVL tree with a right-right violation (Wikipedia)." >}}

The case will be reversed for a left-left violation, where a right rotation is performed.


### Left-Right and Right-Left {#left-right-and-right-left}

The figure below shows a tree with a Right-Left violation. This can happen if either a node was removed from \\(t\_1\\) or a node was added to \\(Y\\), in which case the heights of \\(t\_2\\) and \\(t\_3\\) are not equal. Either way, the right side of the tree needs to be restructured to correct the violation.

{{< figure src="/ox-hugo/2024-11-03_15-58-38_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>An AVL tree with a right-left violation (Wikipedia)." >}}

The first step is to perfrom a right rotation so that \\(Z\\) is a right child of \\(Y\\). This is shown in the figure below. The subtrees themselves are all about the same height, so performing one more rotation will line up the tree correctly.

{{< figure src="/ox-hugo/2024-11-03_16-08-15_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>An AVL tree with a right-left violation after the first rotation (Wikipedia)." >}}

The final step is to perform a left rotation to correct the violation. The tree is now balanced.

{{< figure src="/ox-hugo/2024-11-03_16-08-49_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>An AVL tree with a right-left violation after the second rotation (Wikipedia)." >}}
