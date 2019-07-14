#include "MNIST_Load.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

int reverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

std::vector<std::vector<float>> readMNIST_trainingSet(std::string fileName, int NumberOfImages, int DataOfAnImage) {
	std::vector<std::vector<float>> arr(NumberOfImages, std::vector<float>(DataOfAnImage));
	std::ifstream file(fileName, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)& magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char*)& number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		file.read((char*)& n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		file.read((char*)& n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		for (int i = 0; i < NumberOfImages; ++i) {
			for (int j = 0, l = 0; j < n_rows; ++j)
				for (int k = 0; k < n_cols; ++k, ++l) {
					unsigned char temp = 0;
					file.read((char*)& temp, sizeof(temp));
					arr[i][l] = (float)temp / 1000.0;
				}
			std::cout << i + 1 << " image loaded\r";
		}
		std::cout << std::endl;
	}
	return arr;
}

std::vector<std::vector<float>> readMNIST_labelSet(std::string fileName, int imageNum) {
	std::vector<std::vector<float>> labelSet;
	std::vector<float> label(1, 0);
	std::ifstream fin(fileName, std::ios::binary);
	if (fin.is_open()) {
		unsigned char data(0);
		int n(-1);
		for (int i(0); i < 8; ++i) fin.read((char*)& data, sizeof(data));
		for (int i(0); i < imageNum; ++i) {
			fin.read((char*)& data, sizeof(data));
			labelSet.push_back(label);
			labelSet[i].back() = (int)data;
			std::cout << i + 1 << " label loaded\r";
		}
		std::cout << std::endl;
	}
	fin.close();
	return labelSet;
}

std::vector<std::vector<float>> readMNIST_verifyImage() {
	std::vector<std::vector<float>> data(100, std::vector<float>(784));
	std::string fileName;
	cv::Mat img;
	for (char n('0'), k(0); n <= '9'; ++n)
		for (char m('0'); m <= '9'; ++m, ++k) {
			fileName.append("MNIST\\");
			fileName.push_back(n);
			fileName.push_back('_');
			fileName.push_back(m);
			fileName.append(".png");
			img = cv::imread(fileName, 0);
			if (!img.empty())
				for (int i(0); i < img.rows; ++i)
					for (int j(0); j < img.cols; ++j)
						data[k][i * img.cols + j] = (255.0 - (float)img.at<uchar>(i, j)) / 1000.0;
			fileName.clear();
		}
	return data;
}

std::vector<std::vector<float>> readMNIST_verifyLabel() {
	std::vector<std::vector<float>> labels(100, std::vector<float>(1));
	for (int i(0); i < 100; ++i) labels[i][0] = i / 10;
	return labels;
}