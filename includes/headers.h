#include <stdio.h>
#include <stdlib.h>

typedf struct Edge
{
	int first;
	int second;
	
}Edge;

__global__ void initialize_vertices(Vertex* , int, int );
__global__ void bfs(const Edge* , int* , int* , bool* );