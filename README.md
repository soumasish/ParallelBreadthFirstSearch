#Parallel BFS - Approach I

The implementation of Parallel BFS is based on the following research paper
http://impact.crhc.illinois.edu/shared/papers/effective2010.pdf

~The core algorithm

This algorithm reads an arrray of edges parallely(with one thread being assigned to read each vertex) and writes the hierarchial level of each vertex from the starting vertex in a vertex array.

The algorithm has a quadratic time complexity.