#pragma once
#include "Layer.cuh"

struct FC {
	cublasHandle_t* blasHandle = nullptr;
	cudnnHandle_t* handle = nullptr;
	int batchSize = 1, inputSize = 1, outputSize = 1;

	FC();
	FC(cublasHandle_t*, cudnnHandle_t*, const int, const int, const int);
};

class FullyConnectedLayer : public Layer {
	const int batchSize, inputSize, outputSize;
	cudnnTensorDescriptor_t outputDesc;
	float* weights, * one_vec, * bias, * input, * output, *g_output, * g_weights, * g_bias, * grad;
public:
	FullyConnectedLayer();
	FullyConnectedLayer(cublasHandle_t*, cudnnHandle_t*, const int, const int, const int);
	FullyConnectedLayer(FC);
	~FullyConnectedLayer();
	void forwardProp(float*);
	void backProp(float*);
	void update(const float);
	float* get();
	float* gradient();
	cudnnTensorDescriptor_t descriptor();
private:
	void initialize(cublasHandle_t*, cudnnHandle_t*);
};