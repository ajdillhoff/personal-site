+++
title = "Linked Lists"
authors = ["Alex Dillhoff"]
date = 2023-10-01T00:00:00-05:00
tags = ["computer science", "data structures"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Singly-Linked Lists](#singly-linked-lists)
- [Doubly-Linked Lists](#doubly-linked-lists)
- [Operations](#operations)
- [Exercises](#exercises)

</div>
<!--endtoc-->

A linked list is a **dynamic** and **aggregate** data structure made up of a collection of nodes. The nodes of a linked list can store any data type and are not enforced to contain the same data type. A basic `node` structure may be defined as

```c
struct node {
    void *data;
    struct node *next;
};
```


## Singly-Linked Lists {#singly-linked-lists}

A **singly-linked list**, the most basic form, has a reference to the first node in the list, called the head, and a single link between each node.

{{< figure src="/ox-hugo/2023-10-03_16-52-48_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Diagram of a linked list with 3 nodes. The top sections contain data and the bottom sections contain pointers to the next node." >}}

The definition of a linked list node allows the list to grow dynamically. Nodes can be added at any time to any position in the node, as long as a reference to the node before the new one is known. The last node in a list points to `NULL`.


## Doubly-Linked Lists {#doubly-linked-lists}

More commonly, linked lists are **doubly-linked** in that there is a link to the next node and a link to the previous node. This allows for more flexibility in traversing the list, but requires more memory to store the extra link. A standard implementation will also keep a reference to both the head and the tail of the list. This permits efficient insertion and deletion at both ends of the list.


## Operations {#operations}


### Insertion {#insertion}

A new node can be inserted either at the beginning, the end, or somewhere in between. Inserting at the beginning or end is a constant time operation, but inserting in the middle requires traversing the list to find the correct position. To insert at the beginning, the new node's `next` reference is updated to the old head and the head is updated to the new node.

```python
def insert_at_beginning(head, data):
    new_node = Node(data)
    new_node.next = head
    head = new_node
```

To insert at the end without a reference to the tail, the list must be traversed to find the last node. The new node's `next` reference is set to `NULL` and the last node's `next` reference is set to the new node.

```python
def insert_at_end(head, data):
    new_node = Node(data)
    new_node.next = None
    if head is None:
        head = new_node
    else:
        last_node = head
        while last_node.next is not None:
            last_node = last_node.next
        last_node.next = new_node
```

To insert at the end with a reference to the tail, the new node's `next` reference is set to `NULL` and the tail's `next` reference is set to the new node. The tail is then updated to the new node.

```python
def insert_at_end(head, tail, data):
    new_node = Node(data)
    new_node.next = None
    if head is None:
        head = new_node
        tail = new_node
    else:
        tail.next = new_node
        tail = new_node
```

To insert in the middle, the list must be traversed to find the correct position. The new node's `next` reference is set to the next node and the previous node's `next` reference is set to the new node.

```python
def insert_in_middle(head, data, position):
    new_node = Node(data)
    if head is None:
        head = new_node
    else:
        current_node = head
        for i in range(position - 1):
            current_node = current_node.next
        new_node.next = current_node.next
        current_node.next = new_node
```


### Searching {#searching}

Searching a linked list is a linear time operation. The list is traversed until the desired node is found or the end of the list is reached.

```python
def search(head, data):
    current_node = head
    while current_node is not None:
        if current_node.data == data:
            return current_node
        current_node = current_node.next
    return None
```


### Deletion {#deletion}

Deletion is similar to insertion. A node can be deleted from the beginning, the end, or somewhere in between. Deleting at the beginning or end is a constant time operation, but deleting in the middle requires traversing the list to find the correct position. To delete at the beginning, the head is updated to the second node and the first node is deleted.

```python
def delete_at_beginning(head):
    if head is not None:
        head = head.next
```

To delete at the end without a reference to the tail, the list must be traversed to find the last node. The second to last node's `next` reference is set to `NULL` and the last node is deleted.

```python
def delete_at_end(head):
    if head is not None:
        if head.next is None:
            head = None
        else:
            last_node = head
            while last_node.next.next is not None:
                last_node = last_node.next
            last_node.next = None
```

To delete at the end with a reference to the tail, the tail is updated to the second to last node and the last node is deleted.

```python
def delete_at_end(head, tail):
    if head is not None:
        if head.next is None:
            head = None
            tail = None
        else:
            last_node = head
            while last_node.next.next is not None:
                last_node = last_node.next
            last_node.next = None
            tail = last_node
```

To delete in the middle, the list must be traversed to find the correct position. The previous node's `next` reference is set to the next node and the current node is deleted.

```python
def delete_in_middle(head, position):
    if head is not None:
        current_node = head
        for i in range(position - 1):
            current_node = current_node.next
        current_node.next = current_node.next.next
```


## Exercises {#exercises}

1.  Reverse a singly-linked list in linear time and constant space.
2.  Implement a queue using a linked list.
