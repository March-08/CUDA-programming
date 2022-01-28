
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void query_device() {
    
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("No device found");
    }

    //I hae onli one device
    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);

    printf("Device %d: %s \n", devNo, iProp.name);
    printf("Number of multiprocessors : %d \n", iProp.multiProcessorCount);
    printf("Clock rate %d:  \n", iProp.clockRate);
    printf("Compute capability:  %d.%d:  \n", iProp.major, iProp.minor);
    printf("Total amount of global memory %4.2f KB/n:  \n", iProp.totalGlobalMem/1024.0);
    printf("Total amount of costant memory %4.2f KB/n:  \n", iProp.totalConstMem/1024.0);
    printf("Total amount of shared memory per block %4.2f KB/n:  \n", iProp.sharedMemPerBlock/1024.0);

}



int main(){

    query_device();
    
}
