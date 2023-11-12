#include "NeuralNet.h"
#include <iostream>
#include <cmath>

NeuralNet::NeuralNet(std::initializer_list<Layer> layers) : m_Layers(layers)
{
	size_t _max = 0;
	// Pre-allocate memory for errors
	{
		for (auto& l : m_Layers)
		{
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
		for (int i = 1; i < m_Layers.size(); ++i)
		{
			m_Layers[i - 1].SetConnection(&m_Layers[i]);
		}
	}

}

void NeuralNet::Train(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<double>>& batchOutputs)
{
	assert(m_Layers.size() > 0 && "No layers present!");

	size_t batchSize = batchInputs.size();
	size_t layersSize = m_Layers.size();

	auto& startLayer = m_Layers[0];
	auto& outputLayer = m_Layers.back();

	double error = 0.0;
	// Set gradients to zero
	std::fill(m_Gradient.begin(), m_Gradient.end(), 0);
	
	for (int i = 0; i < batchSize; ++i)
	{
		auto& input = batchInputs[i];
		startLayer.FeedForward(input);
		
		auto& outputs = outputLayer.GetOutputs();

		ComputeErrorGrad(outputs, batchOutputs[i], batchInputs[i]);		
		
		error += ComputeError(outputs, batchOutputs[i]);

	}

	std::cout << "MSE: " << error << "\n";

}

void NeuralNet::ComputeErrorGrad(const std::vector<double>& netOutputs, const std::vector<double>& expctedOutputs, const std::vector<double>& inputs)
{
	size_t size = netOutputs.size();
	for (int i = 0; i < size; ++i)
	{
		m_Gradient[i] = m_Gradient[i] + (netOutputs[i] - expctedOutputs[i]) * inputs[i];
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
