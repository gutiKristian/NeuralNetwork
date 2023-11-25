#pragma once
#include "Activations.h"
#include <cassert>
#include <vector>
#include <iostream>
#include <string>
#include <random>

class Layer
{
	using Matrix = std::vector<std::vector<double>>;
public:
	Layer(size_t In, size_t Size, ActivationFunction activationFunction, ActivationFunction activationPrime) : 
		m_InputSize(In), m_LayerSize(Size), m_ActivationFunc(activationFunction), m_ActivationPrimeFunc(activationPrime)
	{
		m_Weights.resize(m_LayerSize, std::vector<double>(m_InputSize));
		m_Bias.resize(m_LayerSize, 0.0);
		InitWeights();
	};

public:

	/*
	* Input is 2D array of inputs -- batch
	*/
	void Forward(const Matrix& batch)
	{
		assert(batch.size() > 0 && "Batch size must be > 0");
		assert(m_InputSize == batch[0].size()); // assume its unifrom 2D array

		auto batchSize = batch.size();
		
		// Compute for every input in the batch
		for (int k = 0; k < batchSize; ++k)
		{
			// Traverse neuron by neuron
			for (int i = 0; i < m_LayerSize; ++i)
			{
				double current = 0.0;
				for (int j = 0; j < m_InputSize; ++j)
				{
					current += batch[k][j] * m_Weights[i][j];
				}
				current += m_Bias[i];
				m_Potentials[k][i] = current; // Keep potentials for back propagation
				m_Outputs[k][i] = m_ActivationFunc(current);
			}
		}

		if (p_NextLayer != nullptr)
		{
			p_NextLayer->Forward(m_Outputs);
		}
	}

	/*
	* Top layer loss computation. Prepares "input" for the backward pass.
	* @batchPredResult: prediction made by network
	* @batchOutputs: Ground truth
	*/
	void Backward(const Matrix& batchOutputs, bool showLoss)
	{
		assert(batchOutputs.size() == m_Outputs.size() && m_Outputs.size() > 0 && "Size mismatch!");
		assert(batchOutputs[0].size() == m_Outputs[0].size() && m_Outputs[0].size() > 0 && "Size mismatch!");

		auto batchSize = m_Outputs.size();
		auto batchOutputSize = m_Outputs[0].size();

		Matrix gradients;
		gradients.resize(batchSize, std::vector<double>(batchOutputSize, 0.0));

		
		// For MSE -> E = y_i - d_ki
		
		//! For each output
		for (int k = 0; k < batchSize; ++k)
		{
			//! Compute gradient
			for (int i = 0; i < batchOutputSize; ++i)
			{
				gradients[k][i] = m_Outputs[k][i] - batchOutputs[k][i];
			}
		}
		//! 2D arrays of gradients

		Backward(gradients);
	}

	/*
	* @derivedValues: derivation computed in the layer above
	*/
	void Backward(const Matrix& inputDerivation)
	{
		assert(inputDerivation.size() > 0 && "Empty inputDerivation");

		if (p_PrevLayer == nullptr)
		{
			return;
		}

		auto batchSize = inputDerivation.size();
		// Next layer in backpropagation is layer that is previous to this one
		auto nextLayerSize = p_PrevLayer->GetLayerSize();

		Matrix inputNextLayer; // can be preallocated
		inputNextLayer.resize(batchSize, std::vector<double>(nextLayerSize, 0.0));

		// E_k / y_j
		for (int k = 0; k < batchSize; ++k)
		{
			for (int j = 0; j < nextLayerSize; ++j)
			{
				double y_j = 0.0;
				for (int r = 0; r < m_LayerSize; ++r)
				{
					y_j += inputDerivation[k][r] * m_ActivationPrimeFunc(m_Potentials[k][r]) * m_Weights[r][j];
				}
				inputNextLayer[k][j] = y_j;
			}
		}

		// E_k / w_ji
		
		// delete after debug
		auto inputSize = inputDerivation[0].size();
		assert(inputSize == m_LayerSize && "Passed");

		// Update weights
		const Matrix& y_i = p_PrevLayer->GetOutputs();
		for (int k = 0; k < batchSize; ++k)
		{
			for (int i = 0; i < m_LayerSize; ++i)
			{
				for (int j = 0; j < nextLayerSize; ++j)
				{
					m_Weights[i][j] -= m_LearningRate * inputDerivation[k][i] * m_ActivationPrimeFunc(m_Potentials[k][i]) * y_i[k][j];
				}
			}
		}

		// Update biases on this layer
		for (int k = 0; k < batchSize; ++k)
		{
			for (int i = 0; i < m_LayerSize; ++i)
			{
				m_Bias[i] -= m_LearningRate * inputDerivation[k][i] * m_ActivationPrimeFunc(m_Potentials[k][i]);
			}
		}

		p_PrevLayer->Backward(inputNextLayer);
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
		m_Outputs.resize(batchSize, std::vector<double>(m_LayerSize));
		m_Potentials.resize(batchSize, std::vector<double>(m_LayerSize));
	}

	inline const Matrix& GetOutputs() { return m_Outputs; }

	// We do not want to copy the data, Layer class will never outlive the data
	Matrix& SetOutput() { return m_Outputs; }

	inline size_t GetLayerSize() { return m_LayerSize; }

	inline size_t GetInputSize() { return m_InputSize; }

private:
	
	void InitWeights()
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		// Set the mean and standard deviation for the normal distribution
		double mean = 0.0;
		double stddev = 0.001;

		// Create a normal distribution
		std::normal_distribution<double> distribution(mean, stddev);

		for (auto& row : m_Weights)
		{
			for (auto& elem : row)
			{
				elem = distribution(gen);
			}
		}
	}

private:
	// Incoming inputs
	size_t m_InputSize = 0;
	size_t m_LayerSize = 0;
	ActivationFunction m_ActivationFunc;
	ActivationFunction m_ActivationPrimeFunc;

	// Weights between this layer and layer below
	Matrix m_Weights{};
	std::vector<double> m_Bias;
	// Potential and Outputs of current Layer
	Matrix m_Potentials{};
	Matrix m_Outputs{};
	// Backpropagation and learning
	double m_LearningRate = 0.1;
	//
	Layer* p_NextLayer = nullptr;
	Layer* p_PrevLayer = nullptr;
};
