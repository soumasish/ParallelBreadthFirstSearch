#include "../includes/headers.h"

int main(int argc, char** argv){
	
	const int NUM_VERTICES = 1000;
	const int NUM_EDGES = 1000;

	Vertex vertices[NUM_VERTICES];
	Edge edges[NUM_EDGES];
	
	
	bool* h_done;
	bool* d_done;

	&h_done = true;

	while(!h_done){
		cudaMemcpy(&d_done, &h_done, sizeof(bool), cudaHostToDevice);
		//bfs
		cudaMemcpy(&h_done, &d_done, sizeof(bool), cudaHostToDevice);
	}

	

	

	cudaFree(d_done);
	
}