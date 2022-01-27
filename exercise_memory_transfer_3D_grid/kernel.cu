
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cuda.h>

//kernel	
__global__ void access_to_array(int * array, int size ){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	
	if (threadId < size) {
		printf("gid :%d, value : %d\n", threadId, array[threadId]);
	}

}

int main(){

	//array
	int size = 150;
	int byte_size = size * sizeof(int);

	//initialize array
	int* h_array;
	h_array = (int*)malloc(byte_size);

	//randomize seed
	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < size; i++) {
		//random number between 0 and ff = 255
		h_array[i] = (int)rand() & 0xff;
	}

	//move array to cuda memory
	int* d_array;
	cudaMalloc((void**)&d_array, byte_size);

	//transfer, its synchronous, so we wait for the copy to be completed before moving on
	cudaMemcpy(d_array, h_array, byte_size, cudaMemcpyHostToDevice);

	//launche 3d kernel
	dim3 block(2, 2, 2);
	dim3 grid(2,2,2);

	access_to_array << <grid, block >> > (d_array, size);
	cudaDeviceSynchronize();

	free(h_array);
	cudaFree(d_array);

	cudaDeviceReset();
	return 0;
	


}


/*
* 
EQUATIONS FOR GID IN N DIMENSIONS

1D grid of 1D blocks
__device__
int getGlobalIdx_1D_1D(){
return blockIdx.x *blockDim.x + threadIdx.x;
}
1D grid of 2D blocks
__device__
int getGlobalIdx_1D_2D(){
return blockIdx.x * blockDim.x * blockDim.y
 + threadIdx.y * blockDim.x + threadIdx.x;
}
1D grid of 3D blocks
__device__
int getGlobalIdx_1D_3D(){
return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
 + threadIdx.z * blockDim.y * blockDim.x
 + threadIdx.y * blockDim.x + threadIdx.x;
}
2D grid of 1D blocks
__device__ int getGlobalIdx_2D_1D(){
int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int threadId = blockId * blockDim.x + threadIdx.x;
return threadId;
}
2D grid of 2D blocks
__device__
int getGlobalIdx_2D_2D(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int threadId = blockId * (blockDim.x * blockDim.y)
 + (threadIdx.y * blockDim.x) + threadIdx.x;
return threadId;
}
2D grid of 3D blocks
__device__
int getGlobalIdx_2D_3D(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
 + (threadIdx.z * (blockDim.x * blockDim.y))
 + (threadIdx.y * blockDim.x) + threadIdx.x;
return threadId;
}
3D grid of 1D blocks
__device__
int getGlobalIdx_3D_1D(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
 + gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * blockDim.x + threadIdx.x;
return threadId;
}
3D grid of 2D blocks
__device__
int getGlobalIdx_3D_2D(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
 + gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y)
 + (threadIdx.y * blockDim.x) + threadIdx.x;
return threadId;
}
3D grid of 3D blocks
__device__
int getGlobalIdx_3D_3D(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
 + gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
 + (threadIdx.z * (blockDim.x * blockDim.y))
 + (threadIdx.y * blockDim.x) + threadIdx.x;
return threadId;
}
*/