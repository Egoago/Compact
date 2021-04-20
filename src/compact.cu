#include "compact.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void sum(int* pred, int elementCount, int* sum) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    sum[id] = pred[id];
    __syncthreads();
    for (int i = 1; i < elementCount; i *= 2) {
        int tmp = sum[id];
        if (id + i < elementCount)
            sum[id + i] += tmp;
        __syncthreads();
    }
}

__global__ void gather(void* data, int* pred, int* offset, void* result, unsigned int elementSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (pred[id] == 1)
        memcpy((char*)result + (offset[id] - 1) * elementSize, (char*)data + (id * elementSize), elementSize);
}

size_t Compact::operator()(void** compressedData)
{
    cudaMalloc(&dataGPU, elementSize * elementCount);
    cudaMalloc(&predGPU, sizeof(int) * elementCount);
    cudaMalloc(&offsetGPU, sizeof(int) * elementCount);
    cudaMemcpy(dataGPU, dataCPU, elementSize * elementCount, cudaMemcpyHostToDevice);

    predictor << <1, elementCount >> > (dataGPU, predGPU);

    sum << <1, elementCount >> > (predGPU, elementCount, offsetGPU);

    cudaMemcpy(&predCount, &offsetGPU[elementCount - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMalloc(&resultGPU, elementSize * predCount);

    gather << <1, elementCount >> > (dataGPU, predGPU, offsetGPU, resultGPU, elementSize);

    *compressedData = new char[elementSize * predCount];
    cudaMemcpy(*compressedData, resultGPU, elementSize * predCount, cudaMemcpyDeviceToHost);

    cudaFree(dataGPU);
    cudaFree(predGPU);
    cudaFree(offsetGPU);
    cudaFree(resultGPU);
    //cudaDeviceSynchronize();
    return predCount;
}