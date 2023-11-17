#include "NeuralNet.h"
#include <iostream>
#include <cmath>

NeuralNet::NeuralNet(std::initializer_list<Layer> layers, int batchSize) : m_Layers(layers), m_BatchSize(batchSize), 
	m_InputLayer(m_Layers[0].GetLayerSize(), m_Layers[0].GetLayerSize(), nullptr)
{
	size_t _max = 0;
	// Pre-allocate memory
	{
		for (auto& l : m_Layers)
		{
			l.PreAllocateMem(m_BatchSize);
			
			// prob gonna delete this
			if (l.GetLayerSize() > _max)
			{
				_max = l.GetLayerSize();
			}

		}

		m_Gradient.reserve(_max);
		std::fill(m_Gradient.begin(), m_Gradient.end(), 0);
	}

	// Setup connections
	{
		m_Layers[0].SetBackwardConnection(&m_InputLayer);
		for (int i = 1; i < m_Layers.size(); ++i)
		{
			m_Layers[i - 1].SetForwardConnection(&m_Layers[i]);
			m_Layers[i].SetBackwardConnection(&m_Layers[i - 1]);
		}
	}

}

void NeuralNet::Train(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<double>>& batchOutputs)
{
	assert(m_Layers.size() > 0 && "No layers present!");

	auto& startLayer = m_Layers[0];
	auto& outputLayer = m_Layers.back();

	startLayer.FeedForward(batchInputs);

#ifndef _DEBUG
	std::cout << "Here calculate overall loss ?\n";
	// Sum (batch - batchOutput) for every batch in batches and batchOutput in batchOutputs
#endif

	//outputLayer.BackwardPass();

}

void NeuralNet::ComputeErrorGrad(const std::vector<double>& netOutputs, const std::vector<double>& expctedOutputs, const std::vector<double>& inputs)
{
	size_t size = netOutputs.size();
	for (int i = 0; i < size; ++i)
	{
		// MSE
		m_Gradient[i] += (netOutputs[i] - expctedOutputs[i]);
	}
}

double NeuralNet::ComputeError(const std::vector<double>& netOutputs, const std::vector<double>& expectedOutputs)
{
	double error = 0.0;
	size_t size = netOutputs.size();
	for (int i = 0; i < size; ++i)
	{
		error += std::pow(netOutputs[i] - expectedOutputs[i], 2) * 0.5;
	}

	return error;
}
