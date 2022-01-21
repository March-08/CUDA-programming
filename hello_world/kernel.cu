
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//define our kernel 
__global__ void hello_cuda() {
	printf("Hello CUDA\n");
}

int main() {

	int nx, ny;

	nx = 16;
	ny = 4;

	dim3 block(8,2); //its the same as (8,2,1)
	dim3 grid(nx / block.x , ny / block.y);

	//first number of blocks
	//second number of threads per block

	hello_cuda << < block, grid >> > (); //kernel launch asynch
	cudaDeviceSynchronize(); //wait for asynch to finish

	//reset our original device
	cudaDeviceReset();
	return 0;
}