#include "../includes/headers.h"

__global__ void initialize_vertices(int* vertices, int* starting_vertex){
	
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if( v == *starting_vertex){
		vertices[v] = 0;		
	}else{
		vertices[v] = -1;
	} 
}

__global__ void bfs(const Edge* edges, int* vertices, int* current_depth, bool* done){

	int e = blockDim.x * blockIdx.x + threadIdx.x;
	int vfirst = edges[e].first;
	int dfirst = vertices[vfirst];
	int vsecond = edges[e].second;
	int dsecond = vertices[vsecond];

	if((dfirst == *current_depth) && (dsecond == -1)){
		vertices[vsecond] = dfirst +1;
		*current_depth = dfirst+1;
		*done = false;
	}
	if((dsecond == *current_depth) && (dfirst == -1)){
		vertices[vfirst] = dsecond + 1;
		*current_depth = dsecond +1;
		*done = false;
	}
}