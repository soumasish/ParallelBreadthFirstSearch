#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef struct Edge
{
	int first;
	int second;
	
}Edge;

__global__ void initialize_vertices(int* , int);
__global__ void bfs(Edge* , int* , int, int*);