#include "stdio.h"

#include "cuda_runtime_api.h"
int main()
{

	struct cudaDeviceProp prop;
	cudaError_t err = cudaGetDeviceProperties(&prop, 0);

	printf("maxThreadsPerBlock %d\n\r", prop.maxThreadsPerBlock);
	printf("maxThreadsDim %d %d %d\n\r", prop.maxThreadsDim[0],prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize %d %d %d\n\r", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	return 0;
}