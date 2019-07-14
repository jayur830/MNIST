#include "PoolingLayer.cuh"

Pool::Pool() {}

Pool::Pool(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int _batchSize, const int _channels,
	const int _input_height, const int _input_width, const int _stride, cudnnPoolingMode_t _mode) :
	blasHandle(_blasHandle), handle(_handle), batchSize(_batchSize), channels(_channels), input_height(_input_height), input_width(_input_width), stride(_stride), mode(_mode) {}

PoolingLayer::PoolingLayer() : batchSize(1), input_channels(1), output_channels(1),
	input_height(1), input_width(1), mode(CUDNN_POOLING_MAX) {}

PoolingLayer::PoolingLayer(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int _batchSize, const int channels,
	const int _input_height, const int _input_width, const int stride, cudnnPoolingMode_t _mode) :
	batchSize(_batchSize), input_channels(channels), input_height(_input_height), input_width(_input_width), mode(_mode) {
	initialize(_blasHandle, _handle, stride);
}

PoolingLayer::PoolingLayer(Pool& pool) :
	batchSize(pool.batchSize), input_channels(pool.channels), input_height(pool.input_height), input_width(pool.input_width), mode(pool.mode) {
	initialize(pool.blasHandle, pool.handle, pool.stride);
}

PoolingLayer::~PoolingLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputDesc));
	checkCUDNN(cudnnDestroyPoolingDescriptor(this->poolDesc));
	checkCUDA(cudaFree(this->output));
	checkCUDA(cudaFree(this->grad));
}

void PoolingLayer::forwardProp(float* input, cudnnTensorDescriptor_t desc) {
	checkCUDNN(cudnnPoolingForward(*this->handle, this->poolDesc, &this->alpha, this->inputDesc = desc, this->input = input, &this->beta, this->outputDesc, this->output));
}

void PoolingLayer::backProp(float* grad) {
	checkCUDNN(cudnnPoolingBackward(*this->handle, this->poolDesc, &this->alpha, this->outputDesc,
		this->output, this->outputDesc, grad, this->inputDesc, this->input, &this->beta, this->inputDesc, this->grad));
}

float* PoolingLayer::get() {
	return this->output;
}

float* PoolingLayer::gradient() {
	return this->grad;
}

cudnnTensorDescriptor_t PoolingLayer::descriptor() {
	return this->outputDesc;
}

void PoolingLayer::initialize(cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int stride) {
	this->blasHandle = _blasHandle;
	this->handle = _handle;
	checkCUDNN(cudnnCreateTensorDescriptor(&this->inputDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputDesc));
	checkCUDNN(cudnnCreatePoolingDescriptor(&this->poolDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(this->inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->batchSize, this->input_channels, this->input_height, this->input_width));
	checkCUDNN(cudnnSetPooling2dDescriptor(this->poolDesc, this->mode, CUDNN_PROPAGATE_NAN, stride, stride, 0, 0, stride, stride));
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(this->poolDesc, this->inputDesc, &this->batchSize, &this->output_channels, &this->output_height, &this->output_width));
	checkCUDNN(cudnnSetTensor4dDescriptor(this->outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->batchSize, this->output_channels, this->output_height, this->output_width));
	checkCUDA(cudaMalloc(&this->output, sizeof(float) * this->batchSize * this->output_channels * this->output_height * this->output_width));
	checkCUDA(cudaMalloc(&this->grad, sizeof(float) * this->batchSize * this->input_channels * this->input_height * this->input_width));
}