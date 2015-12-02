#include <stdio.h>
#include <stdlib.h>

typedf struct Edge
{
	int first;
	int second;
	
}Edge;

__global__ void initialize_vertices(Vertex* vertices, int starting_vertex, int num_vertices);
__global__ void bfs(const Edge* edges, Vertex* vertices, int current_depth);