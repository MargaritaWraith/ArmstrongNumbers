#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <locale>

cudaError_t Armstrong(int *result, unsigned int size);

__global__ void Kernel(int *result, unsigned int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d \n", i);
	if (i >= size) return;
	int n = i+100;
	int a = n % 10;
	n /= 10;
	int b = n % 10;
	n /= 10;
	int c = n % 10;

	if (a*a*a + b * b*b + c * c*c == i+100)
	{
		result[i] = i+100;
		//printf(" number[%d] = %d \n", i, numbers[i]);
	}
}

int main()
{
	const int arraySize = 900;
	/*int numbers[arraySize];
	for (int i = 0; i < arraySize; i++)
	{
		numbers[i] = 100 + i;
	}*/

	int result[arraySize] = { 0 };

	cudaError_t cudaStatus = Armstrong(result, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	for (int i = 0; i<arraySize; i++)
	{
		if (result[i]!=0)
		{
			printf("%d \n", result[i]);
		}
	}


	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

cudaError_t Armstrong(int *result, unsigned int size)
{
	int *dev_result = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_result, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(dev_result, result, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 block(32, 1);
	dim3 grid((size / 32 + 1), 1);
	Kernel << <grid, block >> > (dev_result, size);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(result, dev_result, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_result);

	return cudaStatus;
}
