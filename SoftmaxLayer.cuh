#pragma once
#include "Layer.cuh"

struct Softmax {
	cublasHandle_t* blasHandle = nullptr;
	cudnnHandle_t* handle = nullptr;
	int batchSize = 1, inputSize = 1, outputSize = 1;

	Softmax();
	Softmax(cublasHandle_t*, cudnnHandle_t*, const int, const int, const int);
};

class SoftmaxLayer : public Layer {
	const int batchSize, inputSize, outputSize;
	cudnnTensorDescriptor_t outputDesc;
	float* input, * output, * grad, loss;
public:
	SoftmaxLayer();
	SoftmaxLayer(cublasHandle_t*, cudnnHandle_t*, const int, const int, const int);
	SoftmaxLayer(Softmax);
	~SoftmaxLayer();
	void forwardProp(float*);
	void backProp(float*);
	float* get();
	float* gradient();
	const float cross_entropy_error(float*);
private:
	void initialize(cublasHandle_t*, cudnnHandle_t*);
};