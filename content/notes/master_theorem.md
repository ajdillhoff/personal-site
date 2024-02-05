+++
title = "Master Theorem"
authors = ["Alex Dillhoff"]
date = 2024-02-04T17:49:00-06:00
tags = ["computer science", "algorithms"]
draft = false
+++

In the study of [Divide and Conquer Algorithms]({{< relref "divide_and_conquer_algorithms.md" >}}), a recurrence tree can be used to determine the runtime complexity. These notes focus on the **master theorem**, a blueprint for solving any recurrence of the form

\\[
T(n) = aT(n/b) + f(n).
\\]
