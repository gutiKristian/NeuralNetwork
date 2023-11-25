#pragma once
#include "Layer.h"
#include <vector>

class NeuralNet
{
public:
	NeuralNet(std::initializer_list<Layer> layers, int batchSize = 1);
public:
	void Train(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<double>>& batchOutputs);

	void Eval(const std::vector< std::vector<double> >& input, const std::vector < std::vector<double> >& output);
	/*
	* Computes error for one input
	*/
	double ComputeError(const std::vector< std::vector<double> >& expectedOutputs);
private:
	std::vector<double> m_Gradient;
	std::vector<Layer> m_Layers;
	Layer m_InputLayer; // Dummy input layer for backprop
	int m_BatchSize;
};