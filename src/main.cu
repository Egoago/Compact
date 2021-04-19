
#include "cuda_runtime.h"

#include <stdio.h>
#include <string.h>
#include "device_launch_parameters.h"
#include "Compact.cuh"

__global__ void prediction(char* data, int* pred) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    pred[id] = (data[id] >= 'f' &&
                data[id] <= 'l');
}

int main()
{
    cudaSetDevice(0);
    const char* abc = "abcdefghijklmnopqrstuvwxyz";

    printf("%s\n", Compact::compress(abc, (unsigned int)strlen(abc), prediction));

    cudaDeviceReset();

    return 0;
}
