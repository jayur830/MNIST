#pragma once
#include <vector>
#include <string>

int reverseInt(int);
std::vector<std::vector<float>> readMNIST_trainingSet(std::string, int, int);
std::vector<std::vector<float>> readMNIST_labelSet(std::string, int);
std::vector<std::vector<float>> readMNIST_verifyImage();
std::vector<std::vector<float>> readMNIST_verifyLabel();