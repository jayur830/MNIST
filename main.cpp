#include "ConvolutionalLayer.cuh"
#include "PoolingLayer.cuh"
#include "FullyConnectedLayer.cuh"
#include "ActivationLayer.cuh"
#include "SoftmaxLayer.cuh"
#include "cuMat.cuh"
#include "MNIST_Load.h"
#include <ctime>

int main() {
	const int batchSize(1), epochSize(15);

	cudnnHandle_t handle;
	cublasHandle_t blasHandle;

	checkCUDNN(cudnnCreate(&handle));
	checkCUBLAS(cublasCreate_v2(&blasHandle));

	Conv c1(&blasHandle, &handle, batchSize, 1, 10, 28, 28, 0, 0, 4, 4, 1, true),
		c2(&blasHandle, &handle, batchSize, 10, 25, 5, 5, 0, 0, 3, 3, 1);
	Pool p1(&blasHandle, &handle, batchSize, 10, 25, 25, 5),
		p2(&blasHandle, &handle, batchSize, 25, 3, 3, 3);
	FC fc1(&blasHandle, &handle, batchSize, 25, 16),
		fc2(&blasHandle, &handle, batchSize, 16, 10);
	Act act(&blasHandle, &handle, batchSize, 16);
	Softmax sm(&blasHandle, &handle, batchSize, 10, 10);

	ConvolutionalLayer conv1(c1);
	PoolingLayer pool1(p1);
	ConvolutionalLayer conv2(c2);
	PoolingLayer pool2(p2);
	FullyConnectedLayer hidden1(fc1);
	ActivationLayer activation(act);
	FullyConnectedLayer hidden2(fc2);
	SoftmaxLayer softmax(sm);

	size_t workspaceSize(0);
	void* workspace(ConvolutionalLayer::setWorkspace(workspaceSize, 2, &conv1, &conv2));

	// Preparing data set
	const int trainImgNum(60000), testImgNum(10000), verifyImgNum(100);

	std::vector<cuMat*>
		trainSet(trainImgNum), labelSet(trainImgNum),
		testSet(testImgNum), testLabelSet(testImgNum),
		verifySet(verifyImgNum), verifyLabels(verifyImgNum);
	std::vector<std::vector<float>>
		_trainSet(readMNIST_trainingSet("train-images.idx3-ubyte", trainImgNum, 784)),
		_labelSet(readMNIST_labelSet("train-labels.idx1-ubyte", trainImgNum)),
		_testSet(readMNIST_trainingSet("t10k-images.idx3-ubyte", testImgNum, 784)),
		_testLabelSet(readMNIST_labelSet("t10k-labels.idx1-ubyte", testImgNum)),
		_verifySet(readMNIST_verifyImage()), _verifyLabelSet(readMNIST_verifyLabel());
	for (int i(0); i < trainImgNum; ++i)
		trainSet[i] = new cuMat(_trainSet[i], 28, 28);
	for (int i(0); i < trainImgNum; ++i) labelSet[i] = new cuMat(_labelSet[i]);
	for (int i(0); i < testImgNum; ++i)
		testSet[i] = new cuMat(_testSet[i], 28, 28);
	for (int i(0); i < testImgNum; ++i) testLabelSet[i] = new cuMat(_testLabelSet[i]);
	for (int i(0); i < verifyImgNum; ++i)
		verifySet[i] = new cuMat(_verifySet[i], 28, 28);
	for (int i(0); i < verifyImgNum; ++i) verifyLabels[i] = new cuMat(_verifyLabelSet[i]);

	std::cout.precision(4);
	std::cout << std::fixed;

	clock_t begin, end;
	cudnnTensorDescriptor_t desc;
	checkCUDNN(cudnnCreateTensorDescriptor(&desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 28, 28));

	// Training
	for (int epoch(1); epoch <= epochSize; ++epoch) {
		std::cout << "<" << epoch << " epoch>" << std::endl;
		double avg(0);
		begin = clock();
		for (int i(0); i < trainImgNum; ++i) {
			std::cout << "\r[" << i + 1 << " iter] ";

			// Forward-propagation
			conv1.forwardProp(trainSet[i]->getMatrix(), desc, workspace, workspaceSize);
			pool1.forwardProp(conv1.get(), conv1.descriptor());
			conv2.forwardProp(pool1.get(), pool1.descriptor(), workspace, workspaceSize);
			pool2.forwardProp(conv2.get(), conv2.descriptor());
			hidden1.forwardProp(pool2.get());
			activation.forwardProp(hidden1.get());
			hidden2.forwardProp(activation.get());
			softmax.forwardProp(hidden2.get());

			// Calculate cost
			int value(-1);
			float data[10], * copy(softmax.get());
			checkCUDA(cudaMemcpy(data, copy, sizeof(float) * 10, cudaMemcpyDeviceToHost));

			float max(0);
			for (int j(0); j < 10; ++j)
				if (max < data[j]) {
					max = data[j];
					value = j;
				}
			if (value == (int)_labelSet[i][0])
				std::cout << "O ";
			else std::cout << "X ";

			const float loss(softmax.cross_entropy_error(labelSet[i]->getMatrix()));
			std::cout << "Loss = " << loss;
			avg += loss;

			// Back-propagation
			softmax.backProp(labelSet[i]->getMatrix());
			hidden2.backProp(softmax.gradient());
			activation.backProp(hidden2.gradient());
			hidden1.backProp(activation.gradient());
			pool2.backProp(hidden1.gradient());
			conv2.backProp(pool2.gradient());
			pool1.backProp(conv2.gradient());
			conv1.backProp(pool1.gradient());

			float learningRate(static_cast<float>(0.01 * pow((1.0 + 0.0001 * i), -0.75)));

			// Weights update
			conv1.update(-learningRate);
			conv2.update(-learningRate);
			hidden1.update(-learningRate);
			hidden2.update(-learningRate);
		}
		end = clock();
		std::cout << "\r[" << trainImgNum << " iter] Average loss = " << avg / trainImgNum;
		std::cout << ", train time : " << (float)(end - begin) / 1000.0 << "ms" << std::endl;
	}

	float acc(0);
	// Test
	for (int i(0); i < testImgNum; ++i) {
		conv1.forwardProp(testSet[i]->getMatrix(), desc, workspace, workspaceSize);
		pool1.forwardProp(conv1.get(), conv1.descriptor());
		conv2.forwardProp(pool1.get(), pool1.descriptor(), workspace, workspaceSize);
		pool2.forwardProp(conv2.get(), conv2.descriptor());
		hidden1.forwardProp(pool2.get());
		activation.forwardProp(hidden1.get());
		hidden2.forwardProp(activation.get());
		softmax.forwardProp(hidden2.get());

		int value(-1);
		float data[10], * copy(softmax.get()), max(0);
		checkCUDA(cudaMemcpy(data, copy, sizeof(float) * 10, cudaMemcpyDeviceToHost));
		std::cout << "input \'" << _testLabelSet[i][0] << "\' => [";
		for (int j(0); j < 10; ++j) {
			std::cout << data[j] << ", ";
			if (max < data[j]) {
				max = data[j];
				value = j;
			}
		}
		std::cout << "\b\b] => predict : " << value;
		if ((int)_testLabelSet[i][0] == value) {
			std::cout << " O" << std::endl;
			acc += 0.01;
		}
		else std::cout << " X" << std::endl;
	}
	std::cout << std::endl << "accuracy = " << acc << "%" << std::endl;

	acc = 0;
	// Verification
	for (int i(0); i < verifyImgNum; ++i) {
		conv1.forwardProp(verifySet[i]->getMatrix(), desc, workspace, workspaceSize);
		pool1.forwardProp(conv1.get(), conv1.descriptor());
		conv2.forwardProp(pool1.get(), pool1.descriptor(), workspace, workspaceSize);
		pool2.forwardProp(conv2.get(), conv2.descriptor());
		hidden1.forwardProp(pool2.get());
		activation.forwardProp(hidden1.get());
		hidden2.forwardProp(activation.get());
		softmax.forwardProp(hidden2.get());

		int value(-1);
		float data[10], * copy(softmax.get()), max(0);
		checkCUDA(cudaMemcpy(data, copy, sizeof(float) * 10, cudaMemcpyDeviceToHost));
		std::cout << "input \'" << _verifyLabelSet[i][0] << "\' => [";
		for (int j(0); j < 10; ++j) {
			std::cout << data[j] << ", ";
			if (max < data[j]) {
				max = data[j];
				value = j;
			}
		}
		std::cout << "\b\b] => predict : " << value;
		if ((int)_verifyLabelSet[i][0] == value) {
			std::cout << " O" << std::endl;
			acc += 1.0;
		}
		else std::cout << " X" << std::endl;
	}
	std::cout << std::endl << "accuracy = " << acc << "%" << std::endl;

	checkCUDNN(cudnnDestroyTensorDescriptor(desc));

	for (int i(0); i < trainImgNum; ++i) {
		delete trainSet[i], labelSet[i];
		trainSet[i] = nullptr;
		labelSet[i] = nullptr;
	}
	for (int i(0); i < testImgNum; ++i) {
		delete testSet[i], testLabelSet[i];
		testSet[i] = nullptr;
		testLabelSet[i] = nullptr;
	}
	for (int i(0); i < verifyImgNum; ++i) {
		delete verifySet[i], verifyLabels[i];
		verifySet[i] = nullptr;
		verifyLabels[i] = nullptr;
	}

	checkCUBLAS(cublasDestroy_v2(blasHandle));
	checkCUDNN(cudnnDestroy(handle));
	system("pause");
	return 0;
}