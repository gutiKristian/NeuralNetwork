#pragma once
#include "Activations.h"
#include <cassert>
#include <vector>
#include <iostream>
#include <string>

class Layer
{
public:
	Layer(size_t In, size_t Size, ActivationFunction activationFunction, std::string name = "Layer") : m_In(In), m_Size(Size), m_ActivationFunction(activationFunction), m_Name(std::move(name))
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
			double current = 0.0;
			for (int j = 0; j < m_In; ++j)
			{
				current += input[j] * m_Weights[i][j];
			}
			m_Potentials[i] = current;
		}

		// Compute outputs, that are forwarded as inputs to the other layer
		for (int i = 0; i < m_Size; ++i)
		{
			m_Outputs[i] = m_ActivationFunction(m_Potentials[i]);
		}

		if (!p_NextLayer)
		{
			p_NextLayer->FeedForward(m_Outputs);
		}
	}

	void SetConnection(Layer* nextLayer)
	{
		p_NextLayer = nextLayer;
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
	std::string m_Name;

	// Weights between this layer and layer below
	std::vector<std::vector<double>> m_Weights{};
	// Potential and Outputs of current Layer
	std::vector<double> m_Potentials{};
	std::vector<double> m_Outputs{};
	//
	Layer* p_NextLayer = nullptr;
};