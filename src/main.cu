
#include "cuda_runtime.h"

#include <stdio.h>
#include <string.h>
#include "device_launch_parameters.h"
#include "Compact.cuh"

__global__ void prediction(char* data, int* pred) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (data[id] > 'f' &&
        data[id] < 'l')
        pred[id] = 1;
    else pred[id] = 0;
}

int main()
{
    cudaSetDevice(0);
    const char* abc = "abcdefghijklmnopqrstuvwxyz";

    printf("%s\n", Compact::compress(abc, strlen(abc), prediction));

    cudaDeviceReset();

    return 0;
}
