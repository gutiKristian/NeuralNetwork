#pragma once

#include "Layer.h"
#include <vector>

class NeuralNet
{
public:
	NeuralNet(std::initializer_list<Layer> layers) : m_Layers(layers) {}
private:
	std::vector<Layer> m_Layers;
};