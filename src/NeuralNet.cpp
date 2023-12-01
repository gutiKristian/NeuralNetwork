#include "NeuralNet.h"
#include <iostream>
#include <cmath>

NeuralNet::NeuralNet(std::initializer_list<Layer> layers, int batchSize) : m_Layers(layers), m_BatchSize(batchSize), 
	m_InputLayer(m_Layers[0].GetInputSize(), m_Layers[0].GetInputSize(), nullptr, nullptr)
{
	size_t _max = 0;
	// Pre-allocate memory
	{
		int index = 0;
		for (auto& l : m_Layers)
		{
			l.SetLayerIndex(index++);
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

	m_Generator = std::mt19937(m_Rd);
}

void NeuralNet::Train(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<int>>& batchOutputs)
{
	assert(m_Layers.size() > 0 && "No layers present!");

	// Generate dropout mask
	std::vector< std::vector<int> > dropoutMask{};

	// Skip output layer
	for (int l = 0; l < m_Layers.size() - 1; ++l)
	{
		dropoutMask.push_back({});
		auto size =  m_Layers[l].GetLayerSize();
		for (int i = 0; i < size; ++i)
		{
			dropoutMask[l].push_back(static_cast<int>(m_Bernoulli(m_Generator)));
		}
	}


	auto& startLayer = m_Layers[0];
	auto& outputLayer = m_Layers.back();
	auto& inputLayerOut = m_InputLayer.SetOutput();
	inputLayerOut = batchInputs;

	startLayer.Forward(batchInputs, dropoutMask);

	outputLayer.Backward(batchOutputs, dropoutMask);

}

double NeuralNet::Eval(const std::vector< std::vector<double> >& input, const std::vector<int>& trueValues)
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

	double acc = 0.0;
	std::cout << "Sample weight: " << lastLayer.GetWeights()[0][0] << "\n";
	acc = (static_cast<double>(hit) / trueValues.size()) * 100;
	std::cout << "Accuracy: " << ((static_cast<double>(hit) / trueValues.size()) * 100) << "%\n";
	return acc;
}

void NeuralNet::AdjustLr(double lr)
{
	m_InputLayer.SetLearningRate(lr);
	for (auto& l : m_Layers)
	{
		l.SetLearningRate(lr);
	}
}

