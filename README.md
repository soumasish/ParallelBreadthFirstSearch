#Parallel BFS using CUDA 7.5

Graph algorithms are a fundamental paradigm of computer Science and is relevant to many domains and application areas. Large graphs involving millions of vertices are common in scientific and engineering applications. Reasonable time bound  implementations of graph algorithms using high-end computing resources have been reported but are accessible to only a few. Graphics Processing Units (GPUs) are fast emerging as inexpensive parallel processors due to their high computation power and low price. The GeForce line of Nvidia GPUs provides the CUDA programming model that treats the GPU as a SIMD processor array. I’ve presented a fundamental graph algorithm - the breadth first search, using this programming model on large graphs. The  results on a graph of 500, 000 vertices would suggest  that the NVIDIA GPUs can be used as a reasonable co-processor to accelerate parts of an application.

Since BFS lends itself well to parallelization there are two common yet distinct strategies that are followed in the parallel execution of BFS: 

	1.	The level-synchronous algorithm.
	2.	The fixed-point algorithm

The level synchronous algorithm uses the following approach; it manages three sets of nodes - the visited set V , the current-level set C, and the next-level set N. Iteratively, the algorithm visits (in parallel) all the nodes in set C and transfers them to set V (in parallel). C is then populated with the nodes from set N, and N is cleared for the new iteration. This iterative process continues until naturally there is no node in the next level. The level synchronous algorithm effectively visits in parallel all nodes in each BFS level, with the parallel execution synchronizing at the end of each level iteration.

The fixed-point algorithm,on the other hand, continuously updates the BFS level of every node, based on BFS levels of all neighboring nodes until no more updates are made. This method is sub-optimal at times because of the lack of communication between neighboring nodes in parallel environments. For the purpose of this project I’ve based my implementation on this approach.



