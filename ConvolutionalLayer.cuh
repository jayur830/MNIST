#pragma once
#include "Layer.cuh"

struct Conv {
	cublasHandle_t* blasHandle = nullptr;
	cudnnHandle_t* handle = nullptr;
	int batchSize = 1;
	int input_channels = 1, output_channels = 1;
	int input_height = 1, input_width = 1;
	int padding_height = 0, padding_width = 0;
	int filter_height = 1, filter_width = 1;
	int stride = 1;
	bool inputLayer = false;

	Conv();
	Conv(cublasHandle_t*, cudnnHandle_t*, const int, const int, const int, const int, 
		const int, const int, const int, const int, const int, const int, const bool isInput = false);
};

class ConvolutionalLayer : public Layer {
	friend class ConvolutionalLayer;
	int batchSize, output_channels, output_height, output_width;
	const int filter_height, filter_width, input_channels, input_height, input_width;
	const bool isInputLayer;
	cudnnTensorDescriptor_t inputDesc, outputDesc, biasDesc;
	cudnnFilterDescriptor_t filterDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t fwdAlgo;
	cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
	cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
	float* input, * output, * weights, * bias, * g_weights, * g_bias, * grad;

public:
	ConvolutionalLayer();
	ConvolutionalLayer(cublasHandle_t*, cudnnHandle_t*, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, const bool isInput = false);
	ConvolutionalLayer(Conv);
	~ConvolutionalLayer();
	static void* setWorkspace(size_t&, const int, ...);
	void forwardProp(float*, cudnnTensorDescriptor_t, void*, size_t);
	void backProp(float*);
	void update(const float);
	float* get();
	float* gradient();
	cudnnTensorDescriptor_t descriptor();
private:
	void initialize(cublasHandle_t*, cudnnHandle_t*, const int, const int, const int);
};