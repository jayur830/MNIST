#pragma once
#include "Layer.cuh"

struct Act {
	cublasHandle_t* blasHandle = nullptr;
	cudnnHandle_t* handle = nullptr;
	cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU;
	int batchSize = 1, inputSize = 1, outputSize = 1;

	Act();
	Act(cublasHandle_t*, cudnnHandle_t*, const int, const int, cudnnActivationMode_t _mode = CUDNN_ACTIVATION_RELU);
};

class ActivationLayer : public Layer {
	const int batchSize, inputSize, outputSize;
	cudnnTensorDescriptor_t outputDesc;
	cudnnActivationDescriptor_t actDesc;
	cudnnActivationMode_t mode;
	float* input, * output, * grad;
public:
	ActivationLayer();
	ActivationLayer(cublasHandle_t*, cudnnHandle_t*, const int, const int, cudnnActivationMode_t _mode = CUDNN_ACTIVATION_RELU);
	ActivationLayer(Act);
	~ActivationLayer();
	void forwardProp(float*);
	void backProp(float*);
	float* get();
	float* gradient();
private:
	void initialize(cublasHandle_t*, cudnnHandle_t*);
};