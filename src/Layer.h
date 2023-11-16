#pragma once
#include "Activations.h"
#include <cassert>
#include <vector>
#include <iostream>
#include <string>

class Layer
{
	using Matrix = std::vector<std::vector<double>>;
public:
	Layer(size_t In, size_t Size, ActivationFunction activationFunction) : m_In(In), m_Size(Size), m_ActivationFunction(activationFunction)
	{
		m_Weights.resize(m_In, std::vector<double>(m_Size));
		Log();
	};

public:

	/*
	* Input is 2D array of inputs -- batch
	*/
	void FeedForward(const Matrix& batches)
	{
		assert(batches.size() > 0 && "Batch size must be > 0");
		assert(m_In == batches[0].size());

		auto batchesSize = batches.size();
		
		// Compute for every input in the batch
		for (int k = 0; k < batchesSize; ++k)
		{
			// Compute inner potential, traverse neuron by neuron
			for (int i = 0; i < m_Size; ++i)
			{
				double current = 0.0;
				for (int j = 0; j < m_In; ++j)
				{
					current += batches[k][j] * m_Weights[i][j];
				}
				m_Potentials[k][i] = current; // Keep potentials for back propagation
				m_Outputs[k][i] = m_ActivationFunction(current);
			}

		}

		if (!p_NextLayer)
		{
			p_NextLayer->FeedForward(m_Outputs);
		}
	}

	// Pass already derived values
	void BackwardPass(const std::vector<double>& derivedValues)
	{
		if (!p_PrevLayer)
		{
			p_PrevLayer->BackwardPass(derivedValues);
		}
	}

	void SetForwardConnection(Layer* nextLayer)
	{
		p_NextLayer = nextLayer;
	}

	void SetBackwardConnection(Layer* previousLayer)
	{
		p_PrevLayer = previousLayer;
	}

	void PreAllocateMem(int batchSize)
	{
		// Using resize on purpose so we can already access with []
		m_Outputs.resize(batchSize, std::vector<double>(m_Size));
		m_Potentials.resize(batchSize, std::vector<double>(m_Size));
	}

	inline const Matrix& GetOutputs() { return m_Outputs; }

	inline size_t GetLayerSize() { return m_Size; }

private:
	void Log()
	{
		std::cout << "Initialized Layer:\n";
		std::cout << "\tInput size : " << m_In << "\n";
		std::cout << "\tLayer size: " << m_Size << "\n";
	}

private:
	// Incoming inputs
	size_t m_In = 0;
	size_t m_Size = 0;
	ActivationFunction m_ActivationFunction;
	// Weights between this layer and layer below
	Matrix m_Weights{};
	// Potential and Outputs of current Layer
	Matrix m_Potentials{};
	Matrix m_Outputs{};
	//
	Layer* p_NextLayer = nullptr;
	Layer* p_PrevLayer = nullptr;
};
