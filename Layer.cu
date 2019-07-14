#include "Layer.cuh"
#include "device_launch_parameters.h"
#include <random>

const char* cublasGetErrorString(cublasStatus_t status) {
	switch (status) {
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "unknown error";
}

Layer::Layer() {
	alpha = 1.0;
	beta = 0;
}

Layer::~Layer() {}

void Layer::xavier(float value, float* weights, int weightSize, float* bias, int biasSize) {
	std::random_device rd;
	std::mt19937 random(rd());
	float w(sqrt(3.0 / value));
	std::uniform_real_distribution<> dist(-w, w);
	for (int i(0); i < weightSize; ++i)
		weights[i] = static_cast<float>(dist(random));
	for (int i(0); i < biasSize; ++i)
		bias[i] = static_cast<float>(dist(random));
}

__global__ void seed_kernel(curandState* state, int seed) {
	int index(blockIdx.x * blockDim.x + threadIdx.x);
	curand_init(seed, index, 0, &state[index]);
}

__global__ void init_kernel(curandState* state, float* arr) {
	int index(blockIdx.x * blockDim.x + threadIdx.x);
	arr[index] = curand_uniform(&state[index]);
}

__global__ void init_one_vec_kernel(float* arr) {
	arr[threadIdx.x] = 1.0;
}