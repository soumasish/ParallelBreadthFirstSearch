#Parallel BFS

The implementation of Parallel BFS is based on the following research paper
http://impact.crhc.illinois.edu/shared/papers/effective2010.pdf

~The core algorithm

This algorithm reads an arrray of edges parallely(with one thread being assigned to read each vertex) and writes the hierarchial level of each vertex from the starting vertex in a vertex array.

The algorithm has a quadratic time complexity, however each thread runs parallely with no data dependency on each other.

~Specs of the NVIDIA device on which the code has been tested

NVIDIA GeForce GT 750M
CUDA version 7.5
Global Memory 2048 MBytes
2 Multiprocessors, 192 CUDA Cores/MP Total 384 CUDA Cores
GPU Max CLock 926 MHz
Memory Clock rate 2508MHz
L2 Cache Size 262144 bytes
Warp Size 32
Maximum number of threads peer multiprocessor 2048
Maximum number of threads per block 1024
Max dimension of thread block(x, y, z) 1024, 1024, 64

