#pragma once
#include "cudnn.h"
#include "cublas.h"
#include "curand_kernel.h"
#include <iostream>

#define checkCUDNN(expression)                               \
{                                                          \
	cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
		std::cerr << "\nError on " << __FILE__  << "/line " << __LINE__ << " : "      \
                << cudnnGetErrorString(status) << std::endl; \
		std::exit(EXIT_FAILURE);                               \
    }                                                        \
}

#define checkCUBLAS(expression)                               \
{                                                          \
	cublasStatus_t status = (expression);                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
		std::cerr << "\nError on " << __FILE__  << "/line " << __LINE__ << " : "      \
                << cublasGetErrorString(status) << std::endl; \
		std::exit(EXIT_FAILURE);                               \
    }                                                        \
}

#define checkCUDA(expression)                               \
{                                                          \
	cudaError_t error = (expression);                     \
    if (error != cudaSuccess) {                    \
		std::cerr << "\nError on " << __FILE__  << "/line " << __LINE__ << " : "      \
                << cudaGetErrorString(error) << std::endl; \
		std::exit(EXIT_FAILURE);                               \
    }                                                        \
}

const char* cublasGetErrorString(cublasStatus_t);

class Layer {
protected:
	cublasHandle_t* blasHandle;
	cudnnHandle_t* handle;
public:
	float alpha, beta;
	void* workspace;
	size_t workspaceSize;

	Layer();
	~Layer();
	void xavier(float, float*, int, float*, int);
};

__global__ void seed_kernel(curandState*, int seed = 0);
__global__ void init_kernel(curandState*, float*);
__global__ void init_one_vec_kernel(float*);