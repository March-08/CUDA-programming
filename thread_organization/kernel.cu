﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void print_details() {
	printf("blockIdx.x : %d, blockIdx.y  : %d, blockIdx.z : %d, blockDim.x : %d, blockDim.y  : %d, blockDim.z : %d, gridDim.x : %d, gridDim.y  : %d, gridDim.z : %d\n",
		blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main() {

	//define grid size
	int nx, ny;
	nx = 16;
	ny = 16;

	//define block size
	dim3 block(8, 8);
	dim3 grid(nx / block.x, ny / block.y); //how many block in grid per dimension

	//launch the kernel
	print_details << <grid, block >> > ();
	cudaDeviceSynchronize(); //to wait the end of the call

	cudaDeviceReset();
	return 0;

}


/*
// Exercise : print details for a 3D grid, wth 4 threads in x,y,z and block size = 2 in each dimension


__global__ void print_details() {
	printf("blockIdx.x : %d, blockIdx.y  : %d, blockIdx.z : %d, blockDim.x : %d, blockDim.y  : %d, blockDim.z : %d, gridDim.x : %d, gridDim.y  : %d, gridDim.z : %d\n",
		blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main() {

	//define grid size
	int nx, ny, nz;
	nx = 4;
	ny = 4;
	nz = 4;

	//define block size
	dim3 block(2, 2, 2);
	dim3 grid(nx / block.x, ny / block.y); //how many block in grid per dimension

	//launch the kernel
	print_details << <grid, block >> > ();
	cudaDeviceSynchronize(); //to wait the end of the call

	cudaDeviceReset();
	return 0;

}
*/