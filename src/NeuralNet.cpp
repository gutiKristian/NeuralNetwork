#include "NeuralNet.h"
#include <iostream>

NeuralNet::NeuralNet(std::initializer_list<Layer> layers) : m_Layers(layers)
{
	// Setup connections
	for (int i = 1; i < m_Layers.size(); ++i)
	{
		m_Layers[i - 1].SetConnection(&m_Layers[i]);
	}
}

void NeuralNet::ComputeForward(const std::vector<std::vector<double>>& batch)
{
	assert(m_Layers.size() > 0 && "No layers present!");

	size_t batchSize = batch.size();
	size_t layersSize = m_Layers.size();

	auto& startLayer = m_Layers[0];
	auto& outputLayer = m_Layers.back();

	for (int i = 0; i < batchSize; ++i)
	{
		auto& input = batch[i];
		startLayer.FeedForward(input);
		// outputLayer do something ?
	}

}
