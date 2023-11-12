#pragma once
#include "Layer.h"
#include <vector>

class NeuralNet
{
public:
	NeuralNet(std::initializer_list<Layer> layers);
public:
	void ComputeForward(const std::vector<std::vector<double>>& batch);
private:
	std::vector<Layer> m_Layers;
};