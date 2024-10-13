+++
title = "Complexity Analysis"
authors = ["Alex Dillhoff"]
date = 2023-09-25T00:00:00-05:00
tags = ["computer science", "algorithms", "complexity analysis"]
draft = false
lastmod = 2024-10-13
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [The notation of complexity analysis](#the-notation-of-complexity-analysis)
- [Formal Definition of Asymptotic Notation](#formal-definition-of-asymptotic-notation)
- [Common Functions](#common-functions)

</div>
<!--endtoc-->



## The notation of complexity analysis {#the-notation-of-complexity-analysis}


### $O$-notation {#o-notation}

$O$-notation, often referred to as "Big Oh" notation, describes an upper bound on the behavior of a function. It really means that the function _will not grow faster_ than the a given rate. This rate is typically the highest-order term in the function, and is often referred to as the "dominant term" or "dominant function".

For example, the function \\(f(n) = 3n^2 + 2n + 1\\) has a dominant term of \\(n^2\\), and so we would say that \\(f(n) = O(n^2)\\). We could also accurately describe \\(f(n)\\) as \\(O(n^3)\\) since it technically does not grow at a faster rate than \\(n^3\\), but this is less common as it misleads the reader into thinking that the function is bounded at \\(n^3\\).


### $&Omega;$-notation {#and-omega-notation}

$&Omega;$-notation is used to describe the lower bound on the asymptotic behavior of a function. Specifically, it means that the function grows _at least as fast_ as the given rate. The function \\(f(n) = 3n^2 + 2n + 1\\) grows at least as fast as \\(n^2\\), so we would say that \\(f(n) = \Omega(n^2)\\). It does not grow as fast as \\(n^3\\), however.

Just like $O$-notation, we can abuse this definition and say that something that grows at least as fast as \\(n^2\\) also grows as fast as \\(n\\). This would lead the reader to believe that the function is bounded at \\(n\\), which is not true. For this reason, we typically use the tightest bound possible.


### $&Theta;$-notation {#and-theta-notation}

$&Theta;$-notation gives a tightly bound characterization of a function's behavior. It gives the rate of growth within a constant factor bounded above as well as constant factor bounded below.

To show that a function is \\(\Theta(f(n))\\), we must show that it is both \\(O(f(n))\\) and \\(\Omega(f(n))\\). Taking our example from above, the function \\(f(n) = 3n^2 + 2n + 1\\) is \\(\Theta(n^2)\\).


### Example: Insertion Sort {#example-insertion-sort}

Let's put this notation to work and characterize the running time of insertion sort. We'll start by writing out the pseudocode for the algorithm:

```python
def insertion_sort(A):
    for i in range(1, len(A)):
        key = A[i]
        j = i - 1
        while i >= 0 and A[j] > key:
            A[j + 1] = A[j]
            j = j - 1
        A[j + 1] = key
```

From our [Introduction to Algorithms]({{< relref "introduction_to_complexity_analysis.md" >}}) lecture, we already know that the outer loop runs \\((n-1)\\) times (although the loop is checked \\(n\\) times). This is not dependent on the order of the \\(n\\) inputs either. The inner loop is dependent on the values of our input. It could run anywhere between 0 and \\(i-1\\) times. In the worst case, we saw that it would run \\(n-1\\) times as well. With this, we concluded that the running time of insertion sort is \\(O(n^2)\\). Since this was derived for the worst-case, it is reasonable to say that insertion sort is \\(O(n^2)\\) for all cases.

The key to the number of operations that the inner loop takes is `A[j + 1] = A[j]`, or the number of times a value is shifted to the right. Given an input of \\(n\\) elements in the worst-case scenario, we can split the input into 3 partitions where the largest \\(\lfloor\frac{n}{4}\rfloor\\) values are in the first partition. The second partition has size \\(\lceil\frac{n}{2}\rceil\\), and the last partition has size \\(\lfloor\frac{n}{4}\rfloor\\). By using the floor and ceiling functions, we can accommodate for odd values of \\(n\\).

When the array is finally sorted, the largest \\(\lfloor\frac{n}{4}\rfloor\\) values will be in the last partition. That means that they would have passed through the middle \\(\lceil\frac{n}{2}\rceil\\) values one at a time. Therefore, we can state that the worst-case is proportional to

\\[
\left(\left\lfloor\frac{n}{4}\right\rfloor\right)\left(\left\lceil\frac{n}{2}\right\rceil\right) \leq \frac{n^2}{8}.
\\]

This is \\(\Omega(n^2)\\), so we can conclude that insertion sort is \\(\Theta(n^2)\\).


### Bonus Example: Selection Sort {#bonus-example-selection-sort}

Use a similar analysis to show that the worst-case for selection sort is \\(\Theta(n^2)\\). As a reminder, selection sort is defined as

```python
def selection_sort(A):
    for i in range(0, len(A)-1):
        min_j = i
        for j in range(i + 1, len(A)):
            if A[j] < A[min_j]:
                min_j = j
        A[i], A[min_j] = A[min_j], A[i]
```

We have already observed that the outer loop iterates \\(n-1\\) times. Even in the best case, the inner loop runs proportional to \\(n\\) times. This is sufficient to conclude that the running time is \\(O(n^2)\\) for all cases.

For showing that the worst case is \\(\Omega(n^2)\\), we could use the same argument as insertion sort. However, that isn't necessary. In _any_ case, the inner loop will run proportional to \\(n\\) times. It is not dependent on any specific arrangement of the input as selection sort is. Therefore, we can conclude that the worst-case is \\(\Omega(n^2)\\), and so selection sort is \\(\Theta(n^2)\\).


## Formal Definition of Asymptotic Notation {#formal-definition-of-asymptotic-notation}

Now that we have established some understanding of the notation, let's define it formally. We typically use functions whose domains are over the set of natural or real numbers.


### $O$-notation {#o-notation}

We previously established that $O$-notation described as **asymptotic upper bound**. It was briefly mentioned that this bound holds within a constant factor, which we will now define more thoroughly. For a function \\(g(n)\\), \\(O(g(n)) = \\{f(n) : \exists c > 0, n\_0 > 0 \text{ such that } 0 \leq f(n) \leq cg(n) \text{ for all } n \geq n\_0\\}\\). It might make more sense to visualize this definition.

{{< figure src="/ox-hugo/2023-09-26_17-43-51_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Visualization of $O$-notation (source: Cormen et al.)" >}}

Notice that the function \\(f(n)\\) is bounded above by \\(cg(n)\\) for all \\(n \geq n\_0\\) in the figure above.

Let's put this definition to the test with an example. Given a function \\(f(n) = 3n^2 + 200n + 1000\\), show that \\(f(n) = O(n^2)\\). The goal is to find positive constants \\(c\\) and \\(n\_0\\) such that \\(3n^2 + 200n + 1000 \leq cn^2\\) for all \\(n \geq n\_0\\). Dividing both sides by \\(n^2\\) yields

\\[
3 + \frac{200}{n} + \frac{1000}{n^2} \leq c.
\\]

This equation has many possible solutions. Let's choose \\(n\_0 = 2\\), then

\\[
3 + \frac{200}{2} + \frac{1000}{2^2} = 3 + 100 + 250 = 353 \leq c.
\\]

Therefore, we can conclude that \\(f(n) = O(n^2)\\).


### $&Omega;$-notation {#and-omega-notation}

The notation used to describe an **asymptotic lower bound** is formally defined as \\(\Omega(g(n)) = \\{f(n) : \exists c > 0, n\_0 > 0 \text{ such that } 0 \leq cg(n) \leq f(n) \text{ for all } n \geq n\_0\\}\\). Again, it is helpful to visualize this.

{{< figure src="/ox-hugo/2023-09-26_18-17-07_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Visualization of $&Omega;$-notation (source: Cormen et al.)" >}}

Notice that the function \\(f(n)\\) is bounded below by \\(cg(n)\\) for all \\(n \geq n\_0\\) in the figure above.

Let's revisit our function from above and show that \\(f(n) = \Omega(n^2)\\). The goal is to find positive constants \\(c\\) and \\(n\_0\\) such that \\(3n^2 + 200n + 1000 \geq cn^2\\) for all \\(n \geq n\_0\\). Dividing both sides by \\(n^2\\) yields

\\[
3 + \frac{200}{n} + \frac{1000}{n^2} \geq c.
\\]

This holds when \\(c = 3\\) and \\(n\_0\\) is any positive integer. To see this, think about what happens to this function as \\(n\\) approaches infinity. The first term will always be 3, and the second and third terms will approach 0. Therefore, we can conclude that \\(f(n) = \Omega(n^2)\\).


### $&Theta;$-notation {#and-theta-notation}

Lastly, the notation used for an **asymptotically tight bound** is \\(\Theta(g(n)) = \\{f(n) : \exists c\_1, c\_2 > 0, n\_0 > 0 \text{ such that } 0 \leq c\_1g(n) \leq f(n) \leq c\_2g(n) \text{ for all } n \geq n\_0\\}\\).

{{< figure src="/ox-hugo/2023-09-26_18-23-25_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Visualization of $&Theta;$-notation (source: Cormen et al.)" >}}

We had mentioned previously that if \\(f(n) = \Omega(g(n))\\) and \\(f(n) = O(g(n))\\), then \\(f(n) = \Theta(g(n))\\). This is formalized in the following theorem, as stated in Cormen et al.

> For any two functions \\(f(n)\\) and \\(g(n)\\), we have \\(f(n) = \Theta(g(n))\\) if and only if \\(f(n) = O(g(n))\\) and \\(f(n) = \Omega(g(n))\\).


### $o$-notation {#o-notation}

There are two less commonly used notations that are worth mentioning. The first is $o$-notation, which is used to describe an upper bound that is _not_ asymptotically tight. It is defined as \\(o(g(n)) = \\{f(n) : \forall c > 0, \exists n\_0 > 0 \text{ such that } 0 \leq f(n) < cg(n) \text{ for all } n \geq n\_0\\}\\).

As an example, the bound on \\(an^3 = O(n^3)\\) is asymptotically tight, but the bound on \\(an^3 = o(n^4)\\) is not. Using the definition of $o$-notation, we can see that \\(an^3 = o(n^4)\\), but \\(an^3 \neq o(n^3)\\).


### $&omega;$-notation {#and-omega-notation}

Analogous to $&Omega;$-notation, $&omega;$-notation is used to describe a lower bound that is _not_ asymptotically tight. It is defined as \\(\omega(g(n)) = \\{f(n) : \forall c > 0, \exists n\_0 > 0 \text{ such that } 0 \leq cg(n) < f(n) \text{ for all } n \geq n\_0\\}\\).

It is true that \\(an^3 = \Omega(n^3)\\), but \\(an^3 \neq \omega(n^3)\\).


### Summary of Notation {#summary-of-notation}

An easy way to remember each notation is to think of the following:

-   $O$-notation: "less than or equal to"
-   $&Omega;$-notation: "greater than or equal to"
-   $&Theta;$-notation: "equal to"
-   $o$-notation: "strictly less than"
-   $&omega;$-notation: "strictly greater than"


### Function Properties {#function-properties}

The following properties are useful when analyzing the asymptotic behavior of functions.


#### Transitivity {#transitivity}

-   If \\(f(n) = O(g(n))\\) and \\(g(n) = O(h(n))\\), then \\(f(n) = O(h(n))\\).
-   If \\(f(n) = \Omega(g(n))\\) and \\(g(n) = \Omega(h(n))\\), then \\(f(n) = \Omega(h(n))\\).
-   If \\(f(n) = \Theta(g(n))\\) and \\(g(n) = \Theta(h(n))\\), then \\(f(n) = \Theta(h(n))\\).


#### Reflexivity {#reflexivity}

-   \\(f(n) = O(f(n))\\)
-   \\(f(n) = \Omega(f(n))\\)
-   \\(f(n) = \Theta(f(n))\\)


#### Symmetry {#symmetry}

-   \\(f(n) = \Theta(g(n))\\) if and only if \\(g(n) = \Theta(f(n))\\).


#### Transpose Symmetry {#transpose-symmetry}

-   \\(f(n) = O(g(n))\\) if and only if \\(g(n) = \Omega(f(n))\\).


## Common Functions {#common-functions}

The functions used to describe both time and space complexity are visualized below.

{{< figure src="/ox-hugo/2023-09-26_19-11-32_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Common functions used in complexity analysis (source: Wikipedia)" >}}
