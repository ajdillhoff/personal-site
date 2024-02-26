+++
title = "Priority Queues"
authors = ["Alex Dillhoff"]
date = 2024-02-24T14:10:00-06:00
tags = ["algorithms", "computer science"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Quick Facts](#quick-facts)
- [Introduction](#introduction)
- [Implementation](#implementation)
- [Exercises](#exercises)

</div>
<!--endtoc-->



## Quick Facts {#quick-facts}

-   ****Time Complexity****: \\(O(\lg n)\\)
-   ****Space Complexity****: \\(O(n)\\)


## Introduction {#introduction}

Besides being the primary data structure for [Heapsort]({{< relref "heapsort.md" >}}), a heap is also used to implement a priority queue. A priority queue is a key-value data structure in which the keys are used to determine the priority of each element in the queue. There are two variants, maximum and minimum, and they support the following operations:

1.  Insert: Add a new element to the queue.
2.  Extract: Remove the element with the maximum/minimum key.
3.  Maximum/Minimum: Return the element with the maximum/minimum key without removing it.
4.  Increase/Decrease Key: Increase or decrease the key of a given element.

You could probably imagine a few use cases for such a queue. For example, a priority queue could be used to schedule tasks in a multitasking operating system. The tasks with the highest priority would be executed first. Another example would be a network router that uses a priority queue to schedule packets for transmission. The packets with the highest priority would be transmitted first. In high performance computing, a priority queue could be used to schedule jobs on a supercomputer. The jobs with the highest priority would be executed first. SLURM is an example of a job scheduler that uses a priority queue.

For simple applications, you could reference your application object directly inside the heap. If the objects themselves are too complex, it is optimal to simply set the _value_ of the heap as a reference to the object. A **handle** is a reference that is added to both the heap and the object; it requires little overhead. This requires that your priority queue update both its own index as well as the object's index as changes are made.

An alternative approach is to establish the map using a hash table. In this case, the priority queue is the only data structure that needs to be updated.


## Implementation {#implementation}

Let us now consider the implementation and analysis of the require operations for a priority queue.


### Extract {#extract}

The _keys_ in a priority queue represent the priority. The _values_ will need to be moved around with them as the priority queue is constructed. Getting the maximum or minimum value is a constant time operation and is executed by returning the first element in the array. To extract the item with the highest priority, the first element is removed and the heap is then heapified.

```python
def max_heap_maximum(A):
    if len(A) < 1:
        raise ValueError("Heap underflow")
    return A[0]

def max_heap_extract_max(A):
    max_val = max_heap_maximum(A)
    A[0] = A[-1]
    A.pop()
    max_heapify(A, 0)
    return max_val
```

As we saw with [Heapsort]({{< relref "heapsort.md" >}}), `max_heapify` runs in \\(O(\lg n)\\) time. The call to `max_heap_extract_max` only adds a few constant operations on top of that, so it runs in \\(O(\lg n)\\) time as well.


### Increase {#increase}

The `max_heap_increase_key` function is used to increase the key of a given element. The function first checks if the new key is less than the current key. If it is, an error is raised. The function then updates the key and then traverses up the heap to ensure that the heap property is maintained.

```python
def max_heap_increase_key(A, obj, key):
    if key < obj.key:
        raise ValueError("New key is smaller than current key")
    obj.key = key
    i = A.index(obj) # gets the index of the object
    while i > 0 and A[parent(i)].key < A[i].key:
        A[i], A[parent(i)] = A[parent(i)], A[i]
        i = parent(i)
```

Moving through the height of the tree is done in \\(O(\lg n)\\) time. Depending on the how the index of the object is found, the complexity could be higher. In most cases, the index is found in constant time.


### Insert {#insert}

The `max_heap_insert` function is used to insert a new element into the heap. The function first appends the new element to the end of the array. It then sets the key of the new element to a very small value and then calls `max_heap_increase_key` to update the key to the correct value.

```python
def max_heap_insert(A, obj, n):
    if len(A) == n:
        raise ValueError("Heap overflow")
    key = float("-inf")
    obj.key = key
    A.append(obj)
    # map obj to the last index -- dependent on the implementation
    max_heap_increase_key(A, obj, key)
```

The call to `max_heap_increase_key` runs in \\(O(\lg n)\\) time, so the `max_heap_insert` function also runs in \\(O(\lg n)\\) time in addition to the time it takes to map the object to its index.


## Exercises {#exercises}

1.  Implement a minimum priority queue.
2.  Implement a decrease key function for a maximum priority queue.
3.  Simulate a job scheduler using a priority queue that considers the priority of the job and the time it was submitted.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
</div>
