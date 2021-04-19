#include "compact.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void* Compact::compress(const char* data, const unsigned int size, void(*pred)(char*, int*))
{
    Compact compact(data, size, pred);
    return compact();
}

char* Compact::operator()()
{
    printf("data: ");
    for (int i = 0; i < size; i++)
        printf("%c ", cpuData[i]);
    printf("\n");

    cudaMalloc(&dataGPU, sizeof(char) * size);
    cudaMemcpy(dataGPU, cpuData, sizeof(char) * size, cudaMemcpyHostToDevice);

    mapPred();
    int* preds = new int(size);
    cudaMemcpy(preds, predGPU, sizeof(int) * size, cudaMemcpyDeviceToHost);
    printf("pred: ");
    for (int i = 0; i < size; i++)
        printf("%d ", preds[i]);
    printf("\n");

    scanSum();
    int predCount = 0;
    cudaMemcpy(&predCount, &offsetGPU[size-1], sizeof(int), cudaMemcpyDeviceToHost);
    int* offsets = new int(size);
    cudaMemcpy(offsets, offsetGPU, sizeof(int) * size, cudaMemcpyDeviceToHost);
    printf("offs: ");
    for (int i = 0; i < size; i++)
        printf("%d ", offsets[i]);
    printf("\n");

    mapGather(predCount);

    char* result = new char(predCount + 1);
    result[predCount] = '\0';
    cudaMemcpy(result, resultGPU, sizeof(char) * predCount, cudaMemcpyDeviceToHost);

    return result;
}

void Compact::mapPred()
{
    cudaMalloc(&predGPU, sizeof(int) * size);
    predictor <<<1, size >>> (dataGPU, predGPU);
}

__global__ void sum(int* pred, int size, int* sum) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    sum[id] = pred[id];
    __syncthreads();
    for (int i = 1; i < size; i *= 2) {
        int tmp = sum[id];
        if (id + i < size)
            sum[id + i] += tmp;
        __syncthreads();
    }
}

void Compact::scanSum()
{
    cudaMalloc(&offsetGPU, sizeof(int) * size);
    sum <<<1, size >>> (predGPU, size, offsetGPU);
}

__global__ void gather(char* data, int* pred, int* offset, char* result) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (pred[id] == 1)
        result[offset[id] - 1] = data[id];
}

void Compact::mapGather(int outsize)
{
    cudaMalloc(&resultGPU, sizeof(char) * outsize);
    gather <<<1, size >>> (dataGPU, predGPU, offsetGPU, resultGPU);
}
