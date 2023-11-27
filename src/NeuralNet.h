#pragma once
#include "Layer.h"
#include <vector>

class NeuralNet
{
public:
	NeuralNet(std::initializer_list<Layer> layers, int batchSize = 1);
public:
	void Train(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<int>>& batchOutputs);

	void Eval(const std::vector< std::vector<double> >& input, const std::vector<int>& trueValues);

private:
	std::vector<double> m_Gradient;
	std::vector<Layer> m_Layers;
	Layer m_InputLayer; // Dummy input layer for backprop
	int m_BatchSize;
};