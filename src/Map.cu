#include "Map.cuh"

__global__ void square(int* dataGPU, int dataSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    dataGPU[index] = dataGPU[index] * dataGPU[index];
}

void map(int* b, const int* a, unsigned int size) {
    int* dataGPU;
    cudaMalloc(&dataGPU, sizeof(int) * size);
    cudaMemcpy(dataGPU, a, sizeof(int) * size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = 4;
    square <<<blocksPerGrid, threadsPerBlock >>> (dataGPU, size);
    cudaMemcpy(b, dataGPU, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaFree(dataGPU);
}