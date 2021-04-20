
#include "cuda_runtime.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "device_launch_parameters.h"
#include "Compact.cuh"

__global__ void evenPred(void* data, int* pred) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    pred[id] = (((int*)data)[id] % 2 == 0);
}

bool evenPredCPU(int data) {
    return data % 2 == 0;
}

int* getEvenNumbers(int* data, size_t size) {
    int* buffer = new int[size];
    int offset = 0;
    for (size_t i = 0; i < size; i++)
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    clock_t cpu_startTime, cpu_endTime;

    const int res = 27;
    printf("______________________________________\n");
    printf("|   size   | gpu ms | cpu ms | wrong |\n");
    for (int s = 0; s < res; s++)
    {
        int size = powf(10,s/3.0f);
        int* numbers = new int[size];
        for (int i = 0; i < size; i++)
            numbers[i] = i;

        float sumElapsedMsGPU = 0;
        float sumElapsedMsCPU= 0;
        int averageCount = 100 * powf(1.0f - s / (float)res,2.0f)+1;
        int wrongCount = 0;
        for (int i = 0; i < averageCount; i++)
        {
            //GPU
            cudaEventRecord(start);
            Compact compact(numbers, sizeof(int), size, evenPred);
            int* evenNumbersGPU = nullptr;
            size_t len = compact((void**)&evenNumbersGPU);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            sumElapsedMsGPU += milliseconds;
            //CPU
            cpu_startTime = clock();
            int* evenNumbersCPU = getEvenNumbers(numbers, size);
            cpu_endTime = clock();
            sumElapsedMsCPU += ((cpu_endTime - cpu_startTime) *1000.0 / CLOCKS_PER_SEC);

            for (int x = 0; x < len; x++)
                if (evenNumbersCPU[x] != evenNumbersGPU[x])
                    wrongCount++;
            delete[] evenNumbersGPU;
            delete[] evenNumbersCPU;
        }
        printf("|%-10d|%-8.3lf|%-8.3lf|%-7.1lf|\n",
            size,
            sumElapsedMsGPU / averageCount,
            sumElapsedMsCPU/ averageCount,
            wrongCount / (float)averageCount);
    }

    
    
    cudaDeviceReset();

    return 0;
}
