#include "Map.cuh"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

__global__ void square(int* dataGPU) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    dataGPU[index] = dataGPU[index] * dataGPU[index];
}

void map(int* b, const int* a, unsigned int size) {
    int* dataGPU;
    cudaMalloc(&dataGPU, sizeof(int) * size);
    cudaMemcpy(dataGPU, a, sizeof(int) * size, cudaMemcpyHostToDevice);
    int threadsPerBlock = size;
    square <<<1, size >>> (dataGPU);
    cudaMemcpy(b, dataGPU, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaFree(dataGPU);
}