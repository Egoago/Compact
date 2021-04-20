#ifndef _COMPACT_
#define _COMPACT_
#include <stdio.h>
class Compact {
private:
    const void* dataCPU;
    const unsigned int elementSize;
    const unsigned int elementCount;
    void* dataGPU = nullptr;
    int* predGPU = nullptr;
    int* offsetGPU = nullptr;
    char* resultGPU = nullptr;
    unsigned int predCount = 0;

    void (*predictor)(void*, int*, unsigned int);

public:
    unsigned int operator()(void** compressedData);

    Compact(const void* data, const unsigned int elementSize, const unsigned int elementCount, void (*pred)(void*, int*, unsigned int))
    : dataCPU(data),
        elementSize(elementSize),
        elementCount(elementCount),
        predictor(pred){}

    ~Compact() {
        cudaError_t cudaStatus;
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "before free: %s\n", cudaGetErrorString(cudaStatus));
        }
        if (dataGPU != nullptr)
            cudaFree(dataGPU);
        if (predGPU != nullptr)
            cudaFree(predGPU);
        if (offsetGPU != nullptr)
            cudaFree(offsetGPU);
        if (resultGPU != nullptr)
            cudaFree(resultGPU);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "free: %s\n", cudaGetErrorString(cudaStatus));
        }
    }
};
#endif