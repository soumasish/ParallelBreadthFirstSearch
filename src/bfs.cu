#include "../includes/headers.h"

__global__ void initialize_vertices(int* vertices, int starting_vertex){
	
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if( v == starting_vertex){
		vertices[v] = 0;		
	}else{
		vertices[v] = -1;
	} 
}

__global__ void bfs(Edge* edges, int* vertices, int current_depth, int* modified){

	int e = blockDim.x * blockIdx.x + threadIdx.x;
	int vfirst = edges[e].first;
	if (vfirst > 1023) {printf("oops %d:%d\n", e, vfirst); return;}
	int dfirst = vertices[vfirst];
	int vsecond = edges[e].second;
	if (vsecond > 1023) {printf("oops %d:%d\n", e, vsecond); return;}
	int dsecond = vertices[vsecond];

	if((dfirst == current_depth) && (dsecond == -1)){
		vertices[vsecond] = current_depth;
		printf("e:%d  depth:%d\n", e, current_depth);
		__syncthreads();
		*modified = 1;
		printf("%d\n", *modified);
	}else if((dsecond == current_depth) && (dfirst == -1)){
		vertices[vfirst] = current_depth;
		printf("e:%d depth:%d\n", e, current_depth);
		__syncthreads();
		*modified = 1;
		printf("%d\n", *modified);
	}
}