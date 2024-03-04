+++
title = "Substitution Method"
authors = ["Alex Dillhoff"]
date = 2024-02-27T19:12:00-06:00
tags = ["algorithms", "computer science"]
draft = false
+++

The **substitution method** is a technique for solving recurrences. It works in two steps:

1.  Guess the solution
2.  Use mathematical induction to verify the guess

This can work very well, especially if we already have some intuition about the problem. Let's start with a simple example: The Tower of Hanoi. In this classic puzzle, we have three pegs and a number of disks of different sizes which can slide onto any peg. The puzzle starts with the disks in a neat stack in ascending order of size on one peg, with the smallest disk on top. The objective is to move the entire stack to another peg, obeying the following rules:

1.  Only one disk can be moved at a time
2.  Each move consists of taking the top disk from one of the stacks and placing it on top of another stack
3.  No disk may be placed on top of a smaller disk

The number of moves required to solve the Tower of Hanoi puzzle is given by the recurrence relation \\(T(n) = 2T(n-1) + 1\\) with the initial condition \\(T(1) = 1\\). We can solve this recurrence using the substitution method.

Our hypothesis might be that \\(T(n) \leq c2^n\\) for all \\(n \geq n\_0\\), where \\(c > 0\\) and \\(n\_0 > 0\\). For the base case, we have \\(T(1) = c \* 2^1 = 1\\), so \\(c = 1/2\\). Now we need to show that \\(T(n) \leq c2^n\\) for all \\(n \geq n\_0\\). We assume that \\(T(k) \leq c2^k\\) for all \\(k < n\\). Then we have:

\begin{align\*}
T(n) &\leq 2T(n-1) + 1 \\\\
&\leq 2\left(\frac{2^{n-1}}{2}\right) + 1 \\\\
&= 2^n - 1 \\\\
\end{align\*}

What if we made a bad guess? Let's try \\(T(n) \leq cn\\) for all \\(n \geq n\_0\\). We have \\(T(1) = c = 1\\), so \\(c = 1\\). Now we need to show that \\(T(n) \leq cn\\) for all \\(n \geq n\_0\\). We assume that \\(T(k) \leq ck\\) for all \\(k < n\\). Then we have:

\begin{align\*}
T(n) &\leq 2T(n-1) + 1 \\\\
&\leq 2c(n-1) + 1 \\\\
&= 2cn - 2c + 1 \\\\
\end{align\*}

This does not work because \\(2cn - 2c + 1 > cn\\) for all \\(c > 1\\). Therefore, our guess was wrong.
