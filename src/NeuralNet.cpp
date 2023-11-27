#include "NeuralNet.h"
#include <iostream>
#include <cmath>

NeuralNet::NeuralNet(std::initializer_list<Layer> layers, int batchSize) : m_Layers(layers), m_BatchSize(batchSize), 
	m_InputLayer(m_Layers[0].GetInputSize(), m_Layers[0].GetInputSize(), nullptr, nullptr)
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

void NeuralNet::Train(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<int>>& batchOutputs)
{
	assert(m_Layers.size() > 0 && "No layers present!");

	auto& startLayer = m_Layers[0];
	auto& outputLayer = m_Layers.back();
	auto& inputLayerOut = m_InputLayer.SetOutput();
	inputLayerOut = batchInputs;

	startLayer.Forward(batchInputs);

	outputLayer.Backward(batchOutputs);

}

void NeuralNet::Eval(const std::vector< std::vector<double> >& input, const std::vector<int>& trueValues)
{
	auto& startLayer = m_Layers[0];
	auto& lastLayer = m_Layers.back();
	int hit = 0;
	
	for (int i = 0; i < input.size(); ++i)
	{
		startLayer.Forward({ input[i] });
		auto& output = lastLayer.GetOutputs();
		int pred = std::distance(output[0].begin(), std::max_element(output[0].begin(), output[0].end()));
		if (pred == trueValues[i]) ++hit;
	}
	std::cout << "Accuracy: " << ((static_cast<double>(hit) / trueValues.size()) * 100) << "%\n";
}

