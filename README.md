# CUDA programming C++
 
The most common deep learning frameworks such as [Tensorflow](https://www.tensorflow.org/) and [PyThorch](https://pytorch.org/) often rely on kernel calls in order to use the GPU to compute parallel computations and accelerate the computation of such networks.
The most famous interface that allows developers to program using the GPU is [CUDA](https://it.wikipedia.org/wiki/CUDA), created by [NVIDIA](https://www.nvidia.com/it-it/).
This repository will keep track of my progress in this area.
I will base it mainly on what I'm learning man by man from my master in deep learning run by [Deep Learning Italia Academy](https://elearning.academy-dli.com/), on [Udemy](https://www.udemy.com/) [CUDA programming Masterclass with C++](https://www.udemy.com/course/cuda-programming-masterclass) and also of course on NVIDIA documentation.

My purpose is to deepen my knowledge about parallel programming!

![parallel_cube](https://developer-blogs.nvidia.com/wp-content/uploads/2012/10/CUDA_Cube_1K.jpg)

 
  
  
## In this repository :
 
- [Hello Wolrd](https://github.com/March-08/CUDA-programming/blob/main/hello_world/kernel.cu)
 
  I learned key concepts such as **host** (cpu) and **device** (gpu) computation, the **context switch** method, and the apparent parallel execution of cpu. The difference between process and thread, how threads share memory. I know that there are 2 level of prallelism **(1) task level** and **(2) data level**. The difference between **parallelism** and **concurrency**. Finally I was able to launch the kernel using the **grid** and **block** parameters
 
 - [Threads Organization](https://github.com/March-08/CUDA-programming/blob/main/thread_organization/kernel.cu)
  
   Often figuring out how and which threads access the kernel function is difficult. I have learned to use variables of type dim3 blockIdx, blockDim, gridDim to identify them. 

 - [Unique Index Calculation](https://github.com/March-08/CUDA-programming/blob/main/unique_index_calculation_using_threadIdx_blockIdx_blockDim/kernel.cu)
  
   Often identifying unique thread IDs can be difficult, especially when using grids and 2 or even 3 dimensional blocks. Here I solve this problem

- [Memory Transfer](https://github.com/March-08/CUDA-programming/blob/main/memory_transfer/kernel.cu)
  
  In addition to processing data on the GPU, we also need to transfer data from the CPU to the GPU, and transfer the results back.


- [Sum Array](https://github.com/March-08/CUDA-programming/tree/main/sum_array)
  
  Let's transfer and sum 2 arrays in GPU. Monitor the time needed using clocks, and also lets handle the CUDA errors creating a macro and wrapping all the CUDA functions.

- [Device Query](https://github.com/March-08/CUDA-programming/tree/main/device_query)
  
  Here is a simple script to query on the fly our device and get its properties
  
 - [Intro to Warps](https://github.com/March-08/CUDA-programming/blob/main/warp_id_thread_id/kernel.cu)
  
    We should consider the parallelism between software and hardware. Since each core of a SM can execute in parallel only a single warp (32 thread) this should be the otimal      number oh threads in a block. If we 1 single thread in a block, the hardware will still assign a warp of 32 with resources for 32 threads, but 31 of htem will be inactive,     and  it will be a waste of resources.
  
  
