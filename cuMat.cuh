#pragma once
#include <vector>

class cuMat {
	float* mat;
	size_t _size;
public:
	cuMat();
	cuMat(float*, int);
	cuMat(float*, int, int);
	cuMat(float**, int, int);
	cuMat(std::vector<float>);
	cuMat(std::vector<float>, int, int);
	cuMat(std::vector<std::vector<float>>);
	~cuMat();
	float* getMatrix(bool gpu = true);
	size_t size();
};