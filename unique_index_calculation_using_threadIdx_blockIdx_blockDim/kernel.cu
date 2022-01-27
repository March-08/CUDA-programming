
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void unique_idx_calc_threadIdx(int *input){
	//each thread access one position of the array
	//but idx in the the 2nd block are the same, we should add offset
	int tid = threadIdx.x;
	printf("threadIdx.x : %d, value : %d \n", tid, input[tid]);

}

//use global value adding offset
__global__ void unique_gid_calculation(int * input) {
	int tid = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int gid = tid + offset;

	printf("blockIdx.x : %d, threadIdx.x : %d, gid : %d, value :%d \n", blockIdx.x, threadIdx.x, gid, input[gid]);

}

__global__ void unique_gid_calculation_2d(int* input) {
	int tid = threadIdx.x;
	int block_offset = blockIdx.x * blockDim.x;
	int row_offset = blockDim.x * gridDim.x * blockIdx.y; 
	int gid = tid + block_offset + row_offset;

	printf("blockIdx.x : %d, blockIdx.y : %d, threadIdx.x : %d, gid : %d, value :%d \n", blockIdx.x, blockIdx.y,threadIdx.x, gid, input[gid]);

}

__global__ void unique_gid_calculation_2d_2d(int* input) {
	int tid = blockDim.x * threadIdx.y + threadIdx. x;

	int num_threads_in_a_block = blockDim.x * blockDim.y; 
	int block_offset = blockIdx.x * num_threads_in_a_block;

	int num_threads_in_a_row = num_threads_in_a_row * gridDim.x;
	int row_offset = num_threads_in_a_row * blockIdx.y;

	int gid = tid + block_offset + row_offset;

	printf("blockIdx.x : %d, blockIdx.y : %d, threadIdx.x : %d, gid : %d, value :%d \n", blockIdx.x, blockIdx.y, threadIdx.x, gid, input[gid]);

}

int main(){
	//launch 8 threads and each trhread access one element of an array and print it

	int array_size = 16;
	int array_byte_size = sizeof(int) * array_size;
	int h_data[] = { 23,9,4,53,65,12,1,33 ,56 ,43, 22 ,45 ,77 ,88, 30, 11,};	//array on host (cpu)

	for (int i = 0; i < array_size; i++) {
		printf("%d ", h_data[i]);
	}
	printf("\n \n");

	//transfer array data on device (gpu)
	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(2,2);
	dim3 grid(2,2);

	unique_gid_calculation_2d << <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;


}