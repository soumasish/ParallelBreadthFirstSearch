#include "../includes/headers.h"

int main(int argc, char** argv){
	
	const int NUM_VERTICES = 10000;
	const int VERTEX_BYTES = NUM_VERTICES * sizeof(int);
	const int NUM_EDGES = 10000;
	const int EDGE_BYTES = NUM_EDGES * sizeof(Edge);
	

	int h_vertices[NUM_VERTICES];

	
	Edge h_edges[NUM_EDGES];
	

	for (int i = 0; i < NUM_VERTICES; ++i)   
	{
	    h_edges[i].first = (rand() % (NUM_VERTICES+1));
	    h_edges[i].second = (rand() % (NUM_VERTICES+1));
	}
	
	Edge* d_edges;
	int* d_vertices;
	int* d_starting_vertex;
	int* h_starting_vertex;
	*h_starting_vertex = 25;

	cudaMalloc((void**)&d_edges, EDGE_BYTES);
	cudaMalloc((void**)&d_vertices, VERTEX_BYTES);

	cudaMemcpy(d_edges, h_edges, EDGE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertices, h_vertices, VERTEX_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_starting_vertex, h_starting_vertex, sizeof(int), cudaMemcpyHostToDevice);

	initialize_vertices<<<10, NUM_VERTICES>>>(d_vertices, d_starting_vertex);
	
	
	bool* h_done;
	bool* d_done;
	int* d_current_depth;
	int* h_current_depth;

	cudaMalloc((void**)&d_done, sizeof(bool));
	cudaMalloc((void**)&d_current_depth, sizeof(int));

	*h_current_depth = 0;
	*h_done = true;
	
	while(!h_done){
		cudaMemcpy(&d_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_current_depth, &h_current_depth, sizeof(int), cudaMemcpyHostToDevice);

		bfs<<<10, NUM_EDGES>>>(h_edges, h_vertices, d_current_depth, d_done);

		cudaMemcpy(&h_done, &d_done, sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(&h_current_depth, &d_current_depth, sizeof(int), cudaMemcpyHostToDevice);
	}

	cudaFree(d_edges);
	cudaFree(d_vertices);
	cudaFree(d_done);
	cudaFree(d_current_depth);
	
	
}