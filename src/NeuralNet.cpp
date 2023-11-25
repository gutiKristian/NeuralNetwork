#include "NeuralNet.h"
#include <iostream>
#include <cmath>

NeuralNet::NeuralNet(std::initializer_list<Layer> layers, int batchSize) : m_Layers(layers), m_BatchSize(batchSize), 
	m_InputLayer(m_Layers[0].GetLayerSize(), m_Layers[0].GetLayerSize(), nullptr, nullptr)
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
	auto& inputLayerOut = m_InputLayer.SetOutput();
	inputLayerOut = batchInputs;

	startLayer.Forward(batchInputs);

#ifndef _DEBUG
	std::cout << "HAHA\n";
	// Sum (batch - batchOutput) for every batch in batches and batchOutput in batchOutputs
#endif

	outputLayer.Backward(batchOutputs, false);

}

void NeuralNet::Eval(const std::vector< std::vector<double> >& input, const std::vector < std::vector<double> > & output)
{
	auto& startLayer = m_Layers[0];
	for (int i = 0; i < output.size(); ++i)
	{
		startLayer.Forward({ input[i] });
		std::cout << input[i][0] << " XOR " << input[i][1] << " is " << m_Layers.back().GetOutputs()[0][0] << "\n";
	}
}


double NeuralNet::ComputeError(const std::vector< std::vector<double> >& expectedOutputs)
{
	const auto& netOutputs = m_Layers.back().GetOutputs();
	double error = 0.0;
	size_t size = netOutputs.size();
	for (int i = 0; i < size; ++i)
	{
		error += std::pow(netOutputs[i][0] - expectedOutputs[i][0], 2) * 0.5;
	}

	return error;
}
