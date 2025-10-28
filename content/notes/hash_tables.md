+++
title = "Hash Tables"
authors = ["Alex Dillhoff"]
date = 2024-03-14T15:16:00-05:00
tags = ["computer science", "algorithms"]
draft = false
lastmod = 2025-10-27
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Hash Functions](#hash-functions)

</div>
<!--endtoc-->

The slides accompanying these lecture notes can be found [here](/teaching/cse5311/lectures/hash_maps.pdf).


## Hash Functions {#hash-functions}

As the name suggests, hash functions play a pivotal role in hash maps, so it is worth investigating what makes a good hash function. With the knowledge that collisions cause additional processing, it would be optimal to avoid them in the first place. **A good hash function should try to avoid getting a collision in the first place.**

The hash function has no way of knowing which buckets are already used within the map, and figuring this out on each call would defeat the purpose of hash maps. If we take on a probabilistic view, it would be ideal for the hash function to distribute the data uniformally. Collisions will still happen, of course, but at least it is less likely that one area of the map will be highly concentrated with values.


### Static Hashing {#static-hashing}

In static hashing, a fixed hash function is used. The only variation is through the distribution of the input keys. Static hashing may sometimes refer to the property of a table that does not resize, but this is less common as many widely used implementations include resizing functions.


#### Division Method {#division-method}

The most naive hash function uses a simple division method. Given a map size of \\(m\\), ensure that any key is mapped to a value in range \\([0, m-1]\\). This is implemented by taking the remainder of division by \\(m\\):

\\[
h(k) = k \text{ mod } m.
\\]

If your input is a string, it should be converted to its ASCII or Unicode values first. Each character could be summed up before applying this division method.

This approach works best in the value of \\(m\\) is not a power of two and is, ideally, a prime number. If \\(m\\) is a power of 2, such as \\(m = 2^p\\), then it only uses the \\(p\\) lowest bits of \\(k\\). If \\(m = 16\\), then \\(p = 4\\), and only the last four bits of \\(k\\) are used to determine the bucket. This can lead to many collisions if the input keys are not uniformly distributed. Choosing a value that is closer to a prime number will guarantee that all bits of \\(k\\) are used in the calculation. This limits the possible sizes of the hash map, which may not be ideal.

See [this stack overflow post](https://stackoverflow.com/questions/5929878/why-is-the-size-127-prime-better-than-128-for-a-hash-table) for more discussion.


#### Multiplication Method {#multiplication-method}

A solution to the limitation of choosing a specific value of \\(m\\) is to use the multiplication method. Given a constant \\(A\\) such that \\(0 < A < 1\\), the hash function is defined as:

\\[
h(k) = \lfloor m (kA \text{ mod } 1) \rfloor.
\\]

This works best when \\(m\\) is a power of 2, as this allows for efficient computation using bit shifts. Even if \\(m\\) is not a power of 2, this method still avoids the limitations of the division method.


#### Multiply-Shift {#multiply-shift}

**Why is choosing \\(m\\) as an exact power of 2 more efficient?**

As a refresher, a bit shift is a low-level operation that shifts all bits in a binary representation of a number to the left or right. This is equivalent to multiplying or dividing the number by 2 for each shift position.


### Random Hashing {#random-hashing}


### Cryptographic Hashing {#cryptographic-hashing}
