#pragma once

#include "Activations.h"
#include <cassert>
#include <vector>
#include <iostream>

class Layer
{
public:
	Layer(size_t In, size_t Size, ActivationFunction activationFunction) : m_In(In), m_Size(Size), m_ActivationFunction(activationFunction)
	{
		m_Outputs.reserve(m_Size);
		m_Potentials.reserve(m_Size);
		m_Weights.reserve(m_Size * m_In);
		Log();
	};

public:

	void FeedForward(const std::vector<double>& input)
	{

		assert(m_In == input.size());

		/*
		* Compute inner potential
		* Traverse neuron by neuron
		*/
		for (int i = 0; i < m_Size; ++i)
		{
			for (int j = 0; j < m_In; ++j)
			{
				m_Potentials[i] += input[j] * m_Weights[i][j];
			}
		}

		// Compute outputs, that are forwarded as inputs to the other layer
		for (int i = 0; i < m_Size; ++i)
		{
			m_Outputs[i] = m_ActivationFunction(m_Potentials[i]);
		}
	}

private:
	void Log()
	{
		std::cout << "Initialized Layer:\n";
		std::cout << "\tInput size : " << m_In << "\n";
		std::cout << "\tLayer size: " << m_Size << "\n";
	}

private:
	size_t m_In = 0;
	size_t m_Size = 0;
	ActivationFunction m_ActivationFunction;

	// Weights between this layer and layer below
	std::vector<std::vector<double>> m_Weights{0.000001};
	// Potential and Outputs of current Layer
	std::vector<double> m_Potentials{0.0};
	std::vector<double> m_Outputs{0.0};
};