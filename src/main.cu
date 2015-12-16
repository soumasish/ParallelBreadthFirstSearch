#include "../includes/headers.h"

int main(int argc, char** argv){
	
	//configurable parameters for data set
	const int NUM_VERTICES = 1024;
	const size_t VERTEX_BYTES = NUM_VERTICES * sizeof(int);
	const int NUM_EDGES = 2048;
	const size_t EDGE_BYTES = NUM_EDGES * sizeof(Edge);
	const int STARTING_VERTEX = 85;
	cudaError_t err = cudaSuccess;
	
	//assign thread configuration
    int threadsPerBlock = 1024;
    int vertexBlocks =(NUM_VERTICES + threadsPerBlock - 1) / threadsPerBlock;
    int edgeBlocks =(NUM_EDGES + threadsPerBlock - 1) / threadsPerBlock;
	clock_t begin, end;
	double time_spent;
	int edgeCounter= 0;
	
	//declare the two arrays on host
	int h_vertices[NUM_VERTICES];
	Edge h_edges[NUM_EDGES];
	
	//read file and write into host array
	FILE *infile;
    const char *path = "DataSet/1024-2048.txt";
    char line[100];
    int first, second;
    infile = fopen(path, "r");

  	if (!infile) {
    	printf("Couldn't open %s for reading\n", path);
    	exit(-1);
  	}
  
	while (fgets(line, sizeof(line), infile)!= NULL) 
	{
		
		sscanf(line, "%d\t%d", &first, &second);

	    h_edges[edgeCounter].first = first;
	    h_edges[edgeCounter].second = second;
	    
	    edgeCounter++;
	}
	
	fclose(infile);

	//debugging log to check that the array has been correctly written
	// for (int i = 0; i < NUM_EDGES; ++i)
	// {
	// 	printf("%d -> %d", h_edges[i].first, h_edges[i].second);
	// 	printf(((i % 4) != 3) ? "\t":"\n");
	// }
	
	
	//define pointers two device arrays
	Edge* d_edges;
	int* d_vertices;

	//allocate memory on device for both arrays
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
	
   
    //copy vertices array from host to device
	err = cudaMemcpy(d_vertices, h_vertices, VERTEX_BYTES, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vertices array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    printf("CUDA kernel launch with %d blocks of %d threads\n", vertexBlocks, threadsPerBlock);

		initialize_vertices<<<vertexBlocks, threadsPerBlock>>>(d_vertices, STARTING_VERTEX);

	err = cudaGetLastError();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch initialization kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	printf("Initialization completed\n");

	err = cudaMemcpy(&h_vertices, d_vertices, VERTEX_BYTES, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vertices array from device to kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //debugging log to check that the vertices has has been correctly initialized and copied back to host
	// for (int i = 0; i < NUM_VERTICES; ++i)
	// {
	// 	printf("%d : %d", i, h_vertices[i]);
	// 	printf(((i % 4) != 3) ? "\t":"\n");
	// }

	err = cudaMemcpy(d_vertices, h_vertices, VERTEX_BYTES, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vertices array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	 //copy edges array from host to device
	err = cudaMemcpy(d_edges, h_edges, EDGE_BYTES, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy edges array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	//Initialize depth counter
	int current_depth = 1;

	//Allocate and initialize termination variable modified on host and device
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

	    printf("CUDA kernel launching with %d blocks of %d threads\n", edgeBlocks, threadsPerBlock);

		bfs<<<edgeBlocks, threadsPerBlock>>>(d_edges, d_vertices, current_depth, d_modified);
		cudaThreadSynchronize();

		err = cudaGetLastError();
		if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to launch bfs kernel (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }
		//printf("Second kernel launch finished\n");

		err = cudaMemcpy(&h_modified, d_modified, sizeof(int), cudaMemcpyDeviceToHost);
		printf("%d\n", h_modified);
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