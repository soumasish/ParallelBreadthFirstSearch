#include "../includes/headers.h"

int main(int argc, char** argv){
	
	const int NUM_VERTICES = 10000;
	const int VERTEX_BYTES = NUM_VERTICES * sizeof(int);
	const int NUM_EDGES = 10000;
	const int EDGE_BYTES = NUM_EDGES * sizeof(Edge);
	const int STARTING_VERTEX = 25;


	int h_vertices[NUM_VERTICES];

	
	for (int i = 0; i < NUM_VERTICES; ++i)
	{
		h_vertices[i] = i;
	}
		
	Edge h_edges[NUM_EDGES];
	
	
	for (int i = 0; i < NUM_VERTICES; ++i)
	{
		Edge* e = malloc(sizeof(Edge));
		e->first = (rand() % (NUM_VERTICES+1));
		e->first = (rand() % (NUM_VERTICES+1));
		memcpy(h_edges[i], e, sizeof(e));
	}	
	
	Edge* d_edges;
	int* d_vertices;

	cudaMalloc((void**)&d_edges, EDGE_BYTES);
	cudaMalloc((void**)&d_vertices, VERTEX_BYTES);

	cudaMemcpy(d_edges, h_edges, EDGE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertices, h_vertices, VERTEX_BYTES, cudaMemcpyHostToDevice);

	initialize_vertices<<<10, NUM_VERTICES>>>(d_vertices, STARTING_VERTEX, NUM_VERTICES);
	
	
	bool* h_done;
	bool* d_done;
	int* d_current_depth;
	int* h_current_depth;

	&h_current_depth = 0;
	&h_done = true;
	
	while(!h_done){
		cudaMemcpy(&d_done, &h_done, sizeof(bool), cudaHostToDevice);
		cudaMemcpy(&d_current_depth, &h_current_depth, sizeof(int), cudaHostToDevice);
		bfs<<<10, NUM_EDGES>>>(h_edges, h_vertices, d_current_depth);
		cudaMemcpy(&h_done, &d_done, sizeof(bool), cudaHostToDevice);
		cudaMemcpy(&h_current_depth, &d_current_depth, sizeof(int), cudaHostToDevice);
	}

	
	cudaFree(d_edges);
	cudaFree(d_vertices);
	cudaFree(d_done);
	
}