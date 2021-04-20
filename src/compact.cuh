#ifndef _COMPACT_
#define _COMPACT_

class Compact {
private:
    const void* dataCPU;
    const size_t elementSize;
    const size_t elementCount;
    void* dataGPU = nullptr;
    int* predGPU = nullptr;
    int* offsetGPU = nullptr;
    char* resultGPU = nullptr;
    size_t predCount = 0;

    void (*predictor)(void*, int*);

public:
    size_t operator()(void** compressedData);

    Compact(const void* data, const size_t elementSize, const size_t elementCount, void (*pred)(void*, int*))
    : dataCPU(data),
        elementSize(elementSize),
        elementCount(elementCount),
        predictor(pred){}

    ~Compact() {
        if (dataGPU != nullptr)
            cudaFree(dataGPU);
        if (predGPU != nullptr)
            cudaFree(predGPU);
        if (offsetGPU != nullptr)
            cudaFree(offsetGPU);
        if (resultGPU != nullptr)
            cudaFree(resultGPU);
    }
};
#endif