#Parallel BFS - Approach I

~The core algorithm

This algorithm reads an arrray of edges parallely(with one thread being assigned to read each vertex) and writes the hierarchial level of each vertex from the starting vertex in a vertex array.

The algorithm has a quadratic time complexity.