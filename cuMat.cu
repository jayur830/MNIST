#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "curand_kernel.h"
#include "cuMat.cuh"
#include <iostream>

cuMat::cuMat() {}

cuMat::cuMat(float* data, int size) : _size(size) {
	cudaMalloc(&this->mat, sizeof(float) * size);
	cudaMemcpy(this->mat, data, sizeof(float) * size, cudaMemcpyHostToDevice);
}

cuMat::cuMat::cuMat(float* data, int height, int width) : _size(height * width) {
	cudaMalloc(&this->mat, sizeof(float) * height * width);
	cudaMemcpy(this->mat, data, sizeof(float) * height * width, cudaMemcpyHostToDevice);
}

cuMat::cuMat(float** data, int height, int width) : _size(height* width) {
	cudaMalloc(&this->mat, sizeof(float) * height * width);
	cudaMemcpy(this->mat, data, sizeof(float) * height * width, cudaMemcpyHostToDevice);
}

cuMat::cuMat(std::vector<float> data) : _size(data.size()) {
	cudaMalloc(&this->mat, sizeof(float) * data.size());
	cudaMemcpy(this->mat, data.data(), sizeof(float) * data.size(), cudaMemcpyHostToDevice);
}

cuMat::cuMat(std::vector<float> data, int height, int width) : _size(height * width) {
	cudaMalloc(&this->mat, sizeof(float) * height * width);
	cudaMemcpy(this->mat, data.data(), sizeof(float) * height * width, cudaMemcpyHostToDevice);
	//float* _data(new float[this->_size]);
	//cudaMemcpy(_data, this->mat, sizeof(float) * this->_size, cudaMemcpyDeviceToHost);
	//for (int i(0); i < height; ++i) {
	//	for (int j(0); j < width; ++j)
	//		std::cout << _data[i * width + j] << " ";
	//	std::cout << std::endl;
	//}
	//delete[] _data;
	//_data = nullptr;
}

cuMat::cuMat(std::vector<std::vector<float>> data) : _size(data.size() * data[0].size()) {
	cudaMalloc(&this->mat, sizeof(float) * data.size() * data[0].size());
	float* temp(new float[data.size() * data[0].size()]);
	for (int i(0); i < data.size(); ++i)
		for (int j(0); j < data[i].size(); ++j)
			temp[i * data[i].size() + j] = data[i][j];
	cudaMemcpy(this->mat, temp, sizeof(float) * data.size() * data[0].size(), cudaMemcpyHostToDevice);
	delete[] temp;
	temp = nullptr;
}

cuMat::~cuMat() {
	cudaFree(this->mat);
}

float* cuMat::getMatrix(bool gpu) {
	if (gpu) return this->mat;
	else {
		float* _data(new float[this->_size]);
		cudaMemcpy(_data, this->mat, sizeof(float) * this->_size, cudaMemcpyDeviceToHost);
		return _data;
	}
}

size_t cuMat::size() {
	return this->_size;
}