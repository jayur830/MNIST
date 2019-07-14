#include "ActivationLayer.cuh"

Act::Act() {}

Act::Act(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int _batchSize, const int _size,
	cudnnActivationMode_t _mode) :
	blasHandle(_blasHandle), handle(_handle), batchSize(_batchSize), inputSize(_size), outputSize(_size), mode(_mode) {}

ActivationLayer::ActivationLayer() :
	batchSize(1), inputSize(1), outputSize(1), mode(CUDNN_ACTIVATION_RELU) {}

ActivationLayer::ActivationLayer(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int _batchSize, const int _size, cudnnActivationMode_t _mode) :
	batchSize(_batchSize), inputSize(_size), outputSize(_size), mode(_mode) {
	initialize(_blasHandle, _handle);
}

ActivationLayer::ActivationLayer(Act act) :
	batchSize(act.batchSize), inputSize(act.inputSize), outputSize(act.outputSize), mode(act.mode) {
	initialize(act.blasHandle, act.handle);
}

ActivationLayer::~ActivationLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputDesc));
	checkCUDNN(cudnnDestroyActivationDescriptor(this->actDesc));

	checkCUDA(cudaFree(this->output));
	checkCUDA(cudaFree(this->grad));
}

void ActivationLayer::forwardProp(float* input) {
	checkCUDNN(cudnnActivationForward(*this->handle, this->actDesc, &this->alpha, this->outputDesc, this->input = input, &this->beta, this->outputDesc, this->output));
}

void ActivationLayer::backProp(float* grad) {
	checkCUDNN(cudnnActivationBackward(*this->handle, this->actDesc, &this->alpha, this->outputDesc,
		this->output, this->outputDesc, grad, this->outputDesc, this->input, &this->beta, this->outputDesc, this->grad));
}

float* ActivationLayer::get() {
	return this->output;
}

float* ActivationLayer::gradient() {
	return this->grad;
}

void ActivationLayer::initialize(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle) {
	this->blasHandle = _blasHandle;
	this->handle = _handle;
	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputDesc));
	checkCUDNN(cudnnCreateActivationDescriptor(&this->actDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(this->outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->batchSize, this->outputSize, 1, 1));
	checkCUDNN(cudnnSetActivationDescriptor(this->actDesc, this->mode, CUDNN_PROPAGATE_NAN, 1.0));
	checkCUDA(cudaMalloc(&this->output, sizeof(float) * this->batchSize * this->outputSize));
	checkCUDA(cudaMalloc(&this->grad, sizeof(float) * this->batchSize * this->inputSize));
}