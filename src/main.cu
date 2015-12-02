#include "../includes/headers.h"

int main(int argc, char** argv){
	
	
	
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