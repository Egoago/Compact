#include "compact.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void sum(int* pred, int elementCount, int* sum) {
    int id = threadIdx.x;
    if(id < elementCount)
        sum[id] = pred[id];
    __syncthreads();
    if (id < elementCount)
        for (int i = 1; i < elementCount; i *= 2) {
            int tmp = sum[id];
            if (id + i < elementCount)
                sum[id + i] += tmp;
            __syncthreads();
        }
}

__global__ void gather(void* data, int* pred, int* offset, void* result, unsigned int elementSize, unsigned int elementCount) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (elementCount > id && pred[id] == 1)
        memcpy((char*)result + (offset[id] - 1) * elementSize, (char*)data + (id * elementSize), elementSize);
}

unsigned int Compact::operator()(void** compressedData)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "beginning: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaMalloc(&dataGPU, elementSize * elementCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc1 failed!\n");
        goto Error;
    }
    cudaStatus = cudaMalloc(&predGPU, sizeof(int) * elementCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc2 failed!\n");
        goto Error;
    }
    cudaStatus = cudaMalloc(&offsetGPU, sizeof(int) * elementCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc3 failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dataGPU, dataCPU, elementSize * elementCount, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy1 failed!\n");
        goto Error;
    }

    predictor <<<(elementCount + 63) / 64, 64 >>> (dataGPU, predGPU, elementCount);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "predictor launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    sum << <1, elementCount>> > (predGPU, elementCount, offsetGPU);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "sum launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaMemcpy(&predCount, &offsetGPU[elementCount - 1], sizeof(int), cudaMemcpyDeviceToHost);

    cudaStatus = cudaMalloc(&resultGPU, elementSize * predCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc4 failed!\n");
        goto Error;
    }

    gather << <(elementCount + 63) / 64, 64 >> > (dataGPU, predGPU, offsetGPU, resultGPU, elementSize, elementCount);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "gather launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    *compressedData = new char[elementSize * predCount];
    cudaStatus = cudaMemcpy(*compressedData, resultGPU, elementSize * predCount, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

Error:
    cudaFree(dataGPU);
    cudaFree(predGPU);
    cudaFree(offsetGPU);
    cudaFree(resultGPU);
    dataGPU = resultGPU = nullptr;
    predGPU = offsetGPU = nullptr;
    return predCount;
}