#include "../includes/headers.h"

int main(int argc, char** argv){
	
	const int NUM_VERTICES = 10240;
	const size_t VERTEX_BYTES = NUM_VERTICES * sizeof(int);
	const int NUM_EDGES = 10240;
	const size_t EDGE_BYTES = NUM_EDGES * sizeof(Edge);
	const int STARTING_VERTEX = 25;
	cudaError_t err = cudaSuccess;

	
	//declare the two arrays on host
	int h_vertices[NUM_VERTICES];
	Edge h_edges[NUM_EDGES];
	

	//fill up the edges array
	for (int i = 0; i < NUM_EDGES; ++i)   
	{
	    h_edges[i].first = (rand() % (NUM_VERTICES+1));
	    h_edges[i].second = (rand() % (NUM_VERTICES+1));
	}
	
	//define the two arrays on the device
	Edge* d_edges;
	int* d_vertices;

	//Allocate memory on device for both arrays
	err = cudaMalloc((void**)&d_edges, EDGE_BYTES);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate edges array on device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMalloc((void**)&d_vertices, VERTEX_BYTES);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate vertices array on device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	

	err = cudaMemcpy(d_edges, h_edges, EDGE_BYTES, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy edges array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_vertices, h_vertices, VERTEX_BYTES, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vertices array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	//assign thread configuration
    int threadsPerBlock = 1024;
    int blocksPerGrid =(NUM_VERTICES + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

		initialize_vertices<<<blocksPerGrid, threadsPerBlock>>>(d_vertices, STARTING_VERTEX);
	err = cudaGetLastError();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch initialization kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	printf("Initialization completed\n");
	
	//Initialize depth counter
	int current_depth = 1;

	//Allocate and initialize done on host and device
	bool* d_done;
	bool h_done;
	err = cudaMalloc((void**)&d_done, sizeof(bool));
	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to allocte d_done(error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}
	

	for(;;){
		if(h_done == true) break;
		h_done = true;
		//printf("Entered while loop\n");
		err = cudaMemcpy(d_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to copy h_done to device(error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    printf("CUDA kernel launching with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

		bfs<<<blocksPerGrid, threadsPerBlock>>>(d_edges, d_vertices, current_depth, d_done);
		cudaThreadSynchronize();

		err = cudaGetLastError();
		if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to launch bfs kernel (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }
		//printf("Second kernel launch finished\n");

		err = cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to copy d_done to host(error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    printf("BFS run for level %d\n", current_depth);
	    current_depth++;


	}
	//printf("Breadth first traversal completed over %d levels\n", h_current_depth);
	cudaFree(d_edges);
	cudaFree(d_vertices);
	//cudaFree(d_done);
	//cudaFree(d_current_depth);
	err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");

	return 0;
	
}