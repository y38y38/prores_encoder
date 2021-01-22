#ifndef __DEBUG_H__
#define __DEBUG_H__

double cpuSecond();

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
        {\
            printf("Error: %s:%d",__FILE__,__LINE__);\
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
        }\
}
#endif

