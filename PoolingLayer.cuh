#pragma once
#include "Layer.cuh"

struct Pool {
	cublasHandle_t* blasHandle = nullptr;
	cudnnHandle_t* handle = nullptr;
	int batchSize = 1, channels = 1;
	int input_height = 1, input_width = 1;
	int stride = 1;
	cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;

	Pool();
	Pool(cublasHandle_t*, cudnnHandle_t*, const int, const int, const int, const int, const int, cudnnPoolingMode_t _mode = CUDNN_POOLING_MAX);
};

class PoolingLayer : public Layer {
	int batchSize, output_channels, output_height, output_width;
	const int input_channels, input_height, input_width;
	cudnnTensorDescriptor_t inputDesc, outputDesc;
	cudnnPoolingDescriptor_t poolDesc;
	cudnnPoolingMode_t mode;
	float* input,* output, * grad;
public:
	PoolingLayer();
	PoolingLayer(cublasHandle_t*, cudnnHandle_t* , const int, const int, const int, const int, const int, cudnnPoolingMode_t mode = CUDNN_POOLING_MAX);
	PoolingLayer(Pool&);
	~PoolingLayer();
	void forwardProp(float*, cudnnTensorDescriptor_t);
	void backProp(float*);
	float* get();
	float* gradient();
	cudnnTensorDescriptor_t descriptor();
private:
	void initialize(cublasHandle_t*, cudnnHandle_t*, const int);
};