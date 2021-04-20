#include "cuda_runtime.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "device_launch_parameters.h"
#include "Compact.cuh"

__global__ void evenPred(void* data, int* pred, unsigned int elementCount) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (elementCount > id)
        pred[id] = (((int*)data)[id] % 2 == 0);
}

bool evenPredCPU(int data) {
    return data % 2 == 0;
}

int* getEvenNumbers(int* data, unsigned int size) {
    int* buffer = new int[size];
    int offset = 0;
    for (unsigned int i = 0; i < size; i++)
    {
        if (evenPredCPU(data[i]))
            buffer[offset++] = data[i];
    }
    int* result = new int[offset];
    memcpy(result, buffer, sizeof(int) * offset);
    delete[] buffer;
    return result;
}

int main()
{
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    clock_t start, end;

    const int res = 10;
    printf(" _______________________________________________\n");
    printf("|   size   | out size | gpu ms | cpu ms | wrong |\n");
    for (int s = 0; s < res; s++)
    {
        int size = (int)powf(10,s/3.0f);
        int* numbers = new int[size];
        for (int i = 0; i < size; i++)
            numbers[i] = i;

        float sumElapsedMsGPU = 0;
        float sumElapsedMsCPU = 0;
        int averageCount = (int)(100.0f * (1.0f - s / (float)res) + 1.0f);
        int wrongCount = 0;
        unsigned int len = 0;
        for (int i = 0; i < averageCount; i++)
        {
            //GPU
            start = clock();
            Compact compact(numbers, sizeof(int), size, evenPred);
            int* evenNumbersGPU = nullptr;
            len = compact((void**)&evenNumbersGPU);
            cudaDeviceSynchronize();
            end = clock();
            sumElapsedMsGPU += (float)((end - start) *1000.0 / CLOCKS_PER_SEC);

            //CPU
            start = clock();
            int* evenNumbersCPU = getEvenNumbers(numbers, size);
            end = clock();
            sumElapsedMsCPU += (float)((end - start) *1000.0 / CLOCKS_PER_SEC);
            for (unsigned int x = 0; x < len; x++)
                if (evenNumbersCPU[x] != evenNumbersGPU[x])
                    wrongCount++;
            delete[] evenNumbersGPU;
            delete[] evenNumbersCPU;
        }

        printf("|%-10d|%-10d|%-8.3lf|%-8.3lf|%-7.1lf|\n",
            size,
            len,
            sumElapsedMsGPU / averageCount,
            sumElapsedMsCPU / averageCount,
            wrongCount / (float)averageCount);
    }

    cudaDeviceReset();

    return 0;
}
