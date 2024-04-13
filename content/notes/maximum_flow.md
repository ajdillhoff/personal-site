+++
title = "Maximum Flow"
authors = ["Alex Dillhoff"]
date = 2024-04-12T18:51:00-05:00
tags = ["algorithms", "computer science"]
draft = false
+++

A flow network is a directed graph in which the edges begin at a node that produces the flow and the adjacent nodes are the ones that receive it. _Flow_ in this context could take on many meanings, such as the amount of water that can flow through a pipe, the amount of data that can be sent through a network, or the amount of traffic that can be sent through a road network. The goal of a flow network is to maximize the flow from the source to the sink.

The problem may have intermediate constraints. For example, a network graph may have a node with limited bandwidth, so the flow through that node must be less than or equal to the bandwidth. These notes review the formal definition of the problem followed by a solution using the Ford-Fulkerson algorithm as well as one related to bipartite matching.
