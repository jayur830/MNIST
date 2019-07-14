#include "device_launch_parameters.h"
#include "SoftmaxLayer.cuh"

Softmax::Softmax() {}

Softmax::Softmax(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int _batchSize, const int _inputSize, const int _outputSize) :
	blasHandle(_blasHandle), handle(_handle), batchSize(_batchSize), inputSize(_inputSize), outputSize(_outputSize) {}

SoftmaxLayer::SoftmaxLayer() : batchSize(1), inputSize(1), outputSize(1) {}

SoftmaxLayer::SoftmaxLayer(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int _batchSize, const int _inputSize, const int _outputSize) :
	batchSize(_batchSize), inputSize(_inputSize), outputSize(_outputSize) {
	initialize(_blasHandle, _handle);
}

SoftmaxLayer::SoftmaxLayer(Softmax softmax) :
	batchSize(softmax.batchSize), inputSize(softmax.inputSize), outputSize(softmax.outputSize) {
	initialize(softmax.blasHandle, softmax.handle);
}

SoftmaxLayer::~SoftmaxLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputDesc));

	checkCUDA(cudaFree(this->output));
	checkCUDA(cudaFree(this->grad));
}

void SoftmaxLayer::forwardProp(float* input) {
	checkCUDNN(cudnnSoftmaxForward(*this->handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		&this->alpha, this->outputDesc, this->input = input, &this->beta, this->outputDesc, this->output));
}

__global__ void loss_kernel(const float* labels, int labelSize, int batchSize, float* loss) {
	int index(blockIdx.x * blockDim.x + threadIdx.x);
	if (index >= batchSize) return;
	const int label(static_cast<int>(labels[index]));
	loss[index * labelSize + label] -= 1.0;
}

void SoftmaxLayer::backProp(float* target) {
	float value(1.0 / static_cast<int>(this->batchSize));
	checkCUDA(cudaMemcpyAsync(this->grad, this->output, sizeof(float) * this->batchSize * this->outputSize, cudaMemcpyDeviceToDevice));
	loss_kernel<<<(this->batchSize + 127) / 128, 128>>>(target, this->outputSize, this->batchSize, this->grad);
	checkCUBLAS(cublasSscal_v2(*this->blasHandle, this->outputSize * this->batchSize, &value, this->grad, 1));
	
}

float* SoftmaxLayer::get() {
	return this->output;
}

float* SoftmaxLayer::gradient() {
	return this->grad;
}

const float SoftmaxLayer::cross_entropy_error(float* target) {
	float* temp(new float[this->batchSize * this->outputSize]), * labels(new float[this->batchSize]);
	checkCUDA(cudaMemcpy(temp, this->output, sizeof(float) * this->batchSize * this->outputSize, cudaMemcpyDeviceToHost));
	checkCUDA(cudaMemcpy(labels, target, sizeof(float) * this->batchSize, cudaMemcpyDeviceToHost));
	this->loss = 0;
	for (int i(0); i < this->batchSize; ++i)
		this->loss += log(temp[i * this->outputSize + (int)labels[i]]);
	delete[] temp, labels;
	temp = nullptr;
	labels = nullptr;
	return -this->loss / this->batchSize;
}

void SoftmaxLayer::initialize(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle) {
	this->blasHandle = _blasHandle;
	this->handle = _handle;
	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(this->outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->batchSize, this->outputSize, 1, 1));
	checkCUDA(cudaMalloc(&this->output, sizeof(float) * this->batchSize * this->outputSize));
	checkCUDA(cudaMalloc(&this->grad, sizeof(float) * this->batchSize * this->inputSize));
}