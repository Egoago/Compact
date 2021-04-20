
#include "cuda_runtime.h"
#include <stdio.h>
#include <string.h>
#include "device_launch_parameters.h"
#include "Compact.cuh"

__global__ void abcPred(void* data, int* pred) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    pred[id] = (((char*)data)[id] >= 'f' &&
                ((char*)data)[id] <= 'l');
}

__global__ void evenPred(void* data, int* pred) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    pred[id] = (((int*)data)[id] % 2 == 0);
}

int main()
{
    cudaSetDevice(0);
    //char demo
    const char* abc = "abcdefghijklmnopqrstuvwxyz";
    Compact compact(abc, sizeof(char), strlen(abc), abcPred);
    char* compressedData = nullptr;
    size_t len = compact((void**)&compressedData);
    for (int i = 0; i < len; i++)
        printf("%c", compressedData[i]);
    printf("\n");

    //int demo
    const int size = 25;
    int* numbers = new int[size];
    for (int i = 0; i < size; i++)
        numbers[i] = i;
    Compact compact2(numbers, sizeof(int), size, evenPred);
    int* evenNumbers = nullptr;
    len = compact2((void**)&evenNumbers);
    for (int i = 0; i < len; i++)
        printf("%d ", evenNumbers[i]);
    printf("\n");
    cudaDeviceReset();

    return 0;
}
