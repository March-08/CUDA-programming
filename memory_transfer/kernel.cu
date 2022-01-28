
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void mem_trs_test(int* input) {

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("tid : %d, gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);
}


//lets handle the situation in which number_0f_threads > array_length
__global__ void mem_trs_test(int* input, int size) {

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size) {
		printf("tid : %d, gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);

	}
}

int main(){
  
	int size = 150;
	int byte_size = size * sizeof(int);

	//usually h stands for "host variable'
	int* h_input;
	//malloc returns a void pointer, so we need to cast to integer
	h_input = (int*)malloc(byte_size);

	//randomly initialize an array, call srand to randomize the seed
	time_t t;
	srand((unsigned)time(&t));
	
	for (int i = 0; i < size; i++) {
		//random value between 0 and 255
		h_input[i] = (int)(rand() & 0xff); 
	}

	//d stands for device variable
	int* d_input;

	//need to allocate memory in device, here we need to cast to double pointer
	cudaMalloc((void**)&d_input, byte_size);

	//transfer initialized array from host to device
	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

	//kernel launch arameters
	dim3 block(32);
	dim3 grid(5);

	mem_trs_test << <grid, block >> > (d_input, size);

	//wait kernel execution to finish
	cudaDeviceSynchronize();

	//reclame memory
	free(h_input);
	cudaFree(d_input);


	cudaDeviceReset();
	return 0;
   
}
