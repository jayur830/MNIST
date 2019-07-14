#include "FullyConnectedLayer.cuh"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

FC::FC() {}

FC::FC(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int _batchSize, const int _inputSize, const int _outputSize) :
	blasHandle(_blasHandle), handle(_handle), batchSize(_batchSize), inputSize(_inputSize), outputSize(_outputSize) {}

FullyConnectedLayer::FullyConnectedLayer() : batchSize(1), inputSize(1), outputSize(1) {}

FullyConnectedLayer::FullyConnectedLayer(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int _batchSize, const int _inputSize, const int _outputSize) :
	batchSize(_batchSize), inputSize(_inputSize), outputSize(_outputSize) {	
	initialize(_blasHandle, _handle);
}

FullyConnectedLayer::FullyConnectedLayer(FC fc) :
	batchSize(fc.batchSize), inputSize(fc.inputSize), outputSize(fc.outputSize) {
	initialize(fc.blasHandle, fc.handle);
}

FullyConnectedLayer::~FullyConnectedLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputDesc));

	checkCUDA(cudaFree(this->weights));
	checkCUDA(cudaFree(this->one_vec));
	checkCUDA(cudaFree(this->bias));
	checkCUDA(cudaFree(this->output));
	checkCUDA(cudaFree(this->g_weights));
	checkCUDA(cudaFree(this->g_bias));
	checkCUDA(cudaFree(this->grad));
}

void FullyConnectedLayer::forwardProp(float* input) {
	checkCUBLAS(cublasSgemm_v2(*this->blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, this->outputSize, this->batchSize, this->inputSize,
		& this->alpha, this->weights, this->inputSize, this->input = input, this->inputSize, & this->beta, this->output, this->outputSize));
	checkCUBLAS(cublasSgemm_v2(*this->blasHandle, CUBLAS_OP_N, CUBLAS_OP_N, this->outputSize, this->batchSize,
		1, & this->alpha, this->bias, this->outputSize, this->one_vec, 1, & this->alpha, this->output, this->outputSize));
}

void FullyConnectedLayer::backProp(float* grad) {
	checkCUBLAS(cublasSgemm_v2(*this->blasHandle, CUBLAS_OP_N, CUBLAS_OP_T, this->inputSize, this->outputSize, this->batchSize,
		&this->alpha, this->input, this->inputSize, grad, this->outputSize, &this->beta, this->g_weights, this->inputSize));
	checkCUBLAS(cublasSgemv_v2(*this->blasHandle, CUBLAS_OP_N, this->outputSize, this->batchSize,
		&this->alpha, grad, this->outputSize, this->one_vec, 1, &this->beta, this->g_bias, 1));
	checkCUBLAS(cublasSgemm_v2(*this->blasHandle, CUBLAS_OP_N, CUBLAS_OP_N, this->inputSize, this->batchSize, this->outputSize,
		&this->alpha, this->weights, this->inputSize, grad, this->outputSize, &this->beta, this->grad, this->inputSize));
}

void FullyConnectedLayer::update(const float learningRate) {
	float learn(learningRate);
	checkCUBLAS(cublasSaxpy_v2(*this->blasHandle, static_cast<int>(this->inputSize * this->outputSize), &learn, this->g_weights, 1, this->weights, 1));
	checkCUBLAS(cublasSaxpy_v2(*this->blasHandle, static_cast<int>(this->inputSize * this->outputSize), &learn, this->g_bias, 1, this->bias, 1));
}

float* FullyConnectedLayer::get() {
	return this->output;
}

float* FullyConnectedLayer::gradient() {
	return this->grad;
}

cudnnTensorDescriptor_t FullyConnectedLayer::descriptor() {
	return this->outputDesc;
}

void FullyConnectedLayer::initialize(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle) {
	this->blasHandle = _blasHandle;
	this->handle = _handle;

	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(this->outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->batchSize, this->outputSize, 1, 1));

	checkCUDA(cudaMalloc(&this->weights, sizeof(float) * this->inputSize * this->outputSize));
	checkCUDA(cudaMalloc(&this->bias, sizeof(float) * this->outputSize));
	checkCUDA(cudaMalloc(&this->one_vec, sizeof(float) * this->batchSize));
	checkCUDA(cudaMalloc(&this->output, sizeof(float) * this->batchSize * this->outputSize));
	checkCUDA(cudaMalloc(&this->g_output, sizeof(float) * this->batchSize * this->outputSize));
	checkCUDA(cudaMalloc(&this->g_weights, sizeof(float) * this->inputSize * this->outputSize));
	checkCUDA(cudaMalloc(&this->g_bias, sizeof(float) * this->outputSize));
	checkCUDA(cudaMalloc(&this->grad, sizeof(float) * this->batchSize * this->inputSize));

	float* w(new float[this->inputSize * this->outputSize]), * b(new float[this->outputSize]);
	xavier(this->inputSize * this->outputSize, w, this->inputSize * this->outputSize, b, this->outputSize);
	checkCUDA(cudaMemcpy(this->weights, w, sizeof(float) * this->inputSize * this->outputSize, cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(this->bias, b, sizeof(float) * this->outputSize, cudaMemcpyHostToDevice));
	delete[] w, b;
	w = nullptr; b = nullptr;
}