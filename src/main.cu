#include "../includes/headers.h"

int main(int argc, char** argv){
	
	const int NUM_VERTICES = 1024;
	const size_t VERTEX_BYTES = NUM_VERTICES * sizeof(int);
	const int NUM_EDGES = 2048;
	const size_t EDGE_BYTES = NUM_EDGES * sizeof(Edge);
	const int STARTING_VERTEX = 82;
	cudaError_t err = cudaSuccess;

	clock_t begin, end;
	double time_spent;

	
	//declare the two arrays on host
	int h_vertices[NUM_VERTICES];
	Edge h_edges[NUM_EDGES];
	

	FILE *infile;
    const char *path = "DataSet/1024.txt";
    char line[100];
    int first, second;
    infile = fopen(path, "r");

  	if (!infile) {
    	printf("Couldn't open %s for reading\n", path);
    	exit(-1);
  	}
  	int i=0;
	while (fgets(line, sizeof(line), infile)!= NULL) 
	{
		
		sscanf(line, "%d\t%d", &first, &second);

	    h_edges[i].first = first;
	    h_edges[i].second = second;
	    i++;
	}
	
	fclose(infile);
	
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
    int blocks = 2;
    //int blocksPerGrid =(NUM_VERTICES + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocks, threadsPerBlock);

		initialize_vertices<<<blocks, threadsPerBlock>>>(d_vertices, STARTING_VERTEX);
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
	int* d_modified;
	int h_modified;
	err = cudaMalloc((void**)&d_modified, sizeof(int));
	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to allocte d_done(error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}
	
	begin = clock();

	do{
		
		h_modified = 0;
		//printf("Entered while loop\n");
		err = cudaMemcpy(d_modified, &h_modified, sizeof(int), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to copy h_done to device(error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    printf("CUDA kernel launching with %d blocks of %d threads\n", blocks, threadsPerBlock);

		bfs<<<blocks, threadsPerBlock>>>(d_edges, d_vertices, current_depth, d_modified);
		cudaThreadSynchronize();

		err = cudaGetLastError();
		if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to launch bfs kernel (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }
		//printf("Second kernel launch finished\n");

		err = cudaMemcpy(&h_modified, d_modified, sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to copy d_done to host(error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    printf("BFS run for level %d\n", current_depth);
	    current_depth++;


	}while(h_modified != 0);
	
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time taken: %f\n", time_spent);
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