#pragma once
#include "Layer.h"
#include <vector>
#include <random>

class NeuralNet
{
public:
	NeuralNet(std::initializer_list<Layer> layers, int batchSize = 1);
public:
	void Train(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<int>>& batchOutputs);

	double Eval(const std::vector< std::vector<double> >& input, const std::vector<int>& trueValues);

	void AdjustLr(double lr);

private:
	std::vector<double> m_Gradient;
	std::vector<Layer> m_Layers;
	Layer m_InputLayer; // Dummy input layer for backprop
	int m_BatchSize;

	std::random_device m_Rd;
	std::mt19937 m_Generator;
	std::bernoulli_distribution m_Bernoulli{0.8}; // 0.5 by default
};