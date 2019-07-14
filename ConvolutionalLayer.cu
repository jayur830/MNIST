#include "ConvolutionalLayer.cuh"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include <cstdarg>

Conv::Conv() {}

Conv::Conv(
	cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int _batchSize, const int _input_channels, const int _output_channels,
	const int height, const int width, const int _padding_height, const int _padding_width, const int _filter_height, const int _filter_width, const int _stride, const bool isInput) :
	blasHandle(_blasHandle), handle(_handle), batchSize(_batchSize), input_channels(_input_channels), output_channels(_output_channels),
	input_height(height), input_width(width), padding_height(_padding_height), padding_width(_padding_width),
	filter_height(_filter_height), filter_width(_filter_width), stride(_stride), inputLayer(isInput) {}

ConvolutionalLayer::ConvolutionalLayer() : batchSize(1), input_channels(1), output_channels(1),
	input_height(1), input_width(1), filter_height(1), filter_width(1), isInputLayer(false) {}

ConvolutionalLayer::ConvolutionalLayer(
	cublasHandle_t* _blasHandle, cudnnHandle_t* _handle, const int _batchSize, const int _input_channels, const int _output_channels,
	const int height, const int width, const int padding_height, const int padding_width, const int _filter_height, const int _filter_width, const int stride, const bool isInput) :
	batchSize(_batchSize), input_channels(_input_channels), output_channels(_output_channels), 
	input_height(height), input_width(width), filter_height(_filter_height), filter_width(_filter_width), isInputLayer(isInput) {
	initialize(_blasHandle, _handle, padding_height, padding_width, stride);
}

ConvolutionalLayer::ConvolutionalLayer(Conv conv) :
	batchSize(conv.batchSize), input_channels(conv.input_channels), output_channels(conv.output_channels),
	input_height(conv.input_height), input_width(conv.input_width), filter_height(conv.filter_height), filter_width(conv.filter_width), isInputLayer(conv.inputLayer) {
	initialize(conv.blasHandle, conv.handle, conv.padding_height, conv.padding_width, conv.stride);
}

ConvolutionalLayer::~ConvolutionalLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(this->biasDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(this->filterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(this->convDesc));
	checkCUDA(cudaFree(this->output));
	checkCUDA(cudaFree(this->weights));
	checkCUDA(cudaFree(this->bias));
	checkCUDA(cudaFree(this->g_weights));
	checkCUDA(cudaFree(this->g_bias));
	checkCUDA(cudaFree(this->grad));
}

void* ConvolutionalLayer::setWorkspace(size_t& workspaceSize, int n, ...) {
	va_list ap;
	va_start(ap, n);
	ConvolutionalLayer* conv;
	for (int i(0); i < n; ++i) {
		size_t size;
		conv = va_arg(ap, ConvolutionalLayer*);
		checkCUDNN(cudnnGetConvolutionForwardAlgorithm(*conv->handle, conv->inputDesc, conv->filterDesc, conv->convDesc, conv->outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv->fwdAlgo));
		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(*conv->handle, conv->inputDesc, conv->filterDesc, conv->convDesc, conv->outputDesc, conv->fwdAlgo, &size));
		workspaceSize = workspaceSize > size ? workspaceSize : size;
		checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(*conv->handle, conv->inputDesc, conv->outputDesc, conv->convDesc, conv->filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &conv->bwdFilterAlgo));
		checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(*conv->handle, conv->inputDesc, conv->outputDesc, conv->convDesc, conv->filterDesc, conv->bwdFilterAlgo, &size));
		workspaceSize = workspaceSize > size ? workspaceSize : size;
		if (!conv->isInputLayer) {
			checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(*conv->handle, conv->filterDesc, conv->outputDesc, conv->convDesc, conv->inputDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &conv->bwdDataAlgo));
			checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(*conv->handle, conv->filterDesc, conv->outputDesc, conv->convDesc, conv->inputDesc, conv->bwdDataAlgo, &size));
			workspaceSize = workspaceSize > size ? workspaceSize : size;
		}
	}
	va_end(ap);
	void* workspace(nullptr);
	if (workspaceSize > 0) checkCUDA(cudaMalloc(&workspace, workspaceSize));
	return workspace;
}

void ConvolutionalLayer::forwardProp(float* input, cudnnTensorDescriptor_t desc, void* workspace, size_t workspaceSize) {
	checkCUDNN(cudnnConvolutionForward(
		*this->handle, &this->alpha, this->inputDesc = desc, this->input = input, this->filterDesc, this->weights, this->convDesc,
		this->fwdAlgo, this->workspace = workspace, this->workspaceSize = workspaceSize, &this->beta, this->outputDesc, this->output));
	checkCUDNN(cudnnAddTensor(*this->handle, &this->alpha, this->biasDesc, this->bias, &this->alpha, this->outputDesc, this->output));
}

void ConvolutionalLayer::backProp(float* grad) {
	checkCUDNN(cudnnConvolutionBackwardBias(*this->handle, &this->alpha, this->outputDesc, grad, &this->beta, this->biasDesc, this->g_bias));
	checkCUDNN(cudnnConvolutionBackwardFilter(*this->handle, &this->alpha, this->inputDesc, this->input, this->outputDesc, grad, this->convDesc, this->bwdFilterAlgo, this->workspace, this->workspaceSize, &this->beta, this->filterDesc, this->g_weights));
	if (!this->isInputLayer)
		checkCUDNN(cudnnConvolutionBackwardData(*this->handle, &this->alpha, this->filterDesc, this->weights, this->outputDesc, grad, this->convDesc, this->bwdDataAlgo, this->workspace, this->workspaceSize, &this->beta, this->inputDesc, this->grad));
}

void ConvolutionalLayer::update(const float learningRate) {
	float learn(learningRate);
	checkCUBLAS(cublasSaxpy_v2(*this->blasHandle, static_cast<int>(this->input_channels * this->filter_height * 
		this->filter_width * this->output_channels), &learn, this->g_weights, 1, this->weights, 1));
	checkCUBLAS(cublasSaxpy_v2(*this->blasHandle, static_cast<int>(this->output_channels),
		&learn, this->g_bias, 1, this->bias, 1));
}

float* ConvolutionalLayer::get() {
	return this->output;
}

float* ConvolutionalLayer::gradient() {
	return this->grad;
}

cudnnTensorDescriptor_t ConvolutionalLayer::descriptor() {
	return this->outputDesc;
}

void ConvolutionalLayer::initialize(cublasHandle_t* blasHandle, cudnnHandle_t* handle, const int padding_height, const int padding_width, const int stride) {
	this->blasHandle = blasHandle;
	this->handle = handle;
	checkCUDNN(cudnnCreateTensorDescriptor(&this->inputDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&this->filterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&this->convDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&this->biasDesc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(this->convDesc, padding_height, padding_width, stride, stride, 1, 1,
		CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	checkCUDNN(cudnnSetTensor4dDescriptor(this->inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		this->batchSize, this->input_channels, this->input_height, this->input_width));
	checkCUDNN(cudnnSetFilter4dDescriptor(this->filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
		this->output_channels, this->input_channels, this->filter_height, this->filter_width));
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(this->convDesc, this->inputDesc, this->filterDesc,
		&this->batchSize, &this->output_channels, &this->output_height, &this->output_width));
	checkCUDNN(cudnnSetTensor4dDescriptor(this->outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		this->batchSize, this->output_channels, this->output_height, this->output_width));
	checkCUDNN(cudnnSetTensor4dDescriptor(this->biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		1, this->output_channels, 1, 1));	
	checkCUDA(cudaMalloc(&this->output, sizeof(float) * this->batchSize * this->output_channels * this->output_height * this->output_width));
	checkCUDA(cudaMalloc(&this->weights, sizeof(float) * this->input_channels * this->output_channels * this->filter_height * this->filter_width));
	checkCUDA(cudaMalloc(&this->bias, sizeof(float) * this->output_channels));
	checkCUDA(cudaMalloc(&this->g_weights, sizeof(float) * this->input_channels * this->output_channels * this->filter_height * this->filter_width));
	checkCUDA(cudaMalloc(&this->g_bias, sizeof(float) * this->output_channels));
	checkCUDA(cudaMalloc(&this->grad, sizeof(float) * this->batchSize * this->input_channels * this->input_height * this->input_width));

	float* w(new float[this->input_channels * this->output_channels * this->filter_height * this->filter_width]), * b(new float[this->output_channels]);
	xavier(this->input_channels * this->output_channels * this->filter_height * this->filter_width * this->input_channels, w, this->input_channels * this->output_channels * this->filter_height * this->filter_width, b, this->output_channels);
	checkCUDA(cudaMemcpy(this->weights, w, sizeof(float) * this->input_channels * this->output_channels * this->filter_height * this->filter_width, cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(this->bias, b, sizeof(float) * this->output_channels, cudaMemcpyHostToDevice));
	delete[] w, b;
	w = nullptr; b = nullptr;
}