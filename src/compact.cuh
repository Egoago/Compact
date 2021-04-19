#ifndef _COMPACT_
#define _COMPACT_

class Compact {
private:
    const char* cpuData;
    unsigned int size;
    char* dataGPU = nullptr;
    int* predGPU = nullptr;
    int* offsetGPU = nullptr;
    char* resultGPU = nullptr;

    void (*predictor)(char*, int*);

    void mapPred();
    void scanSum();
    void mapGather(int);

    char* operator()();
public:
    static char* compress(const char* data, const unsigned int size, void (*pred)(char*, int*));

    Compact(const char* data, const unsigned int size, void (*pred)(char*, int*))
    : cpuData(data),size(size),predictor(pred){}

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