#include "../includes/headers.h"

__global__ void initialize_vertices(int* vertices, int starting_vertex){
	
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if( v == starting_vertex){
		vertices[v] = 0;		
	}else{
		vertices[v] = -1;
	} 
}

__global__ void bfs(Edge* edges, int* vertices, int previous_depth, int current_depth, int* modified){

	int e = blockDim.x * blockIdx.x + threadIdx.x;
	int vfirst = edges[e].first;
	
	int dfirst = vertices[vfirst];

	int vsecond = edges[e].second;
	
	int dsecond = vertices[vsecond];

	if((dfirst == previous_depth) && (dsecond == -1)){
		vertices[vsecond] = current_depth;
		__syncthreads();
		*modified = 1;
		
	}else if((dsecond == previous_depth) && (dfirst == -1)){
		vertices[vfirst] = current_depth;
		__syncthreads();
		*modified = 1;
		
	}
}