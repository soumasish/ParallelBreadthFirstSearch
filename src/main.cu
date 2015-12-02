#include "../includes/headers.h"

int main(int argc, char** argv){
	
	const int NUM_VERTICES = 1000;
	const int NUM_EDGES = 1000;
	bool createEdgeArray = false;

	Vertex vertices[NUM_VERTICES];
	Edge edges[NUM_EDGES];
	
	if(createEdgeArray == false){
		for (int i = 0; i < NUM_VERTICES; ++i)
		{
			Edge* e = malloc(sizeof(Edge));
			e->first = (rand() % (NUM_VERTICES+1));
			e->first = (rand() % (NUM_VERTICES+1));
			memcpy(edges[i], e, sizeof(e));
		}
		createEdgeArray = true;
	}
	
	
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