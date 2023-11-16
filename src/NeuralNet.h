#pragma once
#include "Layer.h"
#include <vector>

class NeuralNet
{
public:
	NeuralNet(std::initializer_list<Layer> layers, int batchSize = 1);
public:
	void Train(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<double>>& batchOutputs);

	/*
	* Accumulates gradient error for batch
	*/
	void ComputeErrorGrad(const std::vector<double>& netOutputs, const std::vector<double>& expctedOutputs, const std::vector<double>& inputs);
	/*
	* Computes error for one input
	*/
	double ComputeError(const std::vector<double>& netOutputs, const std::vector<double>& expctedOutputs);
private:
	std::vector<double> m_Gradient;
	std::vector<Layer> m_Layers;
	int m_BatchSize;
};