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
	Layer(size_t In, size_t Size, ActivationFunction activationFunction) : m_InputSize(In), m_LayerSize(Size), m_ActivationFunc(activationFunction)
	{
		m_Weights.resize(m_InputSize, std::vector<double>(m_LayerSize));
		m_Bias.resize(m_LayerSize, 0.0);
		Log();
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

		if (!p_NextLayer)
		{
			p_NextLayer->Forward(m_Outputs);
		}
	}

	/*
	* Top layer loss computation.
	* @batchPredResult: prediction made by network
	* @batchOutputs: true values
	*/
	void BackwardPass(const Matrix& batchPredResult, const Matrix& batchOutputs)
	{
		assert(batchOutputs.size() == batchPredResult.size() && batchPredResult.size() > 0 && "Size mismatch!");
		assert(batchOutputs[0].size() == batchPredResult[0].size() && batchPredResult[0].size() > 0 && "Size mismatch!");

		//! BACKWARD PASS FOR OUTPUT LAYER. ITS SORT OF PREPARING INPUT FOR THE BACKWARD PASS
		std::vector<double> gradients(batchOutputs[0].size(), 0); // can be preallocated

		auto batchSize = batchPredResult.size();
		auto batchOutputSize = batchPredResult[0].size();

		// For MSE -> E = sum(y_i - d_ki)
		for (int i = 0; i < batchSize; ++i)
		{
			for (int j = 0; j < batchOutputSize; ++j)
			{
				gradients[j] += (batchPredResult[i][j] - batchOutputs[i][j]);
			}
		}

		for (int i = 0; i < batchOutputSize; ++i)
		{
			gradients[i] /= batchSize;
		}

		//! BACKWARD PASS FOR HIDDEN LAYER, IT USES THE REC. FORMULA, ALSO TO UPDATE WEIGHTS BETWEEN OUTPUT AND LAST HIDDEN
		//! WE USE PRECOMPUTED VALUE FROM OUTPUT AND SUBSTITUTE IT INTO FORMULA WHERE WE ALSO USE WEIGHTS THAT BELONG TO THIS LAYER

		std::vector<double> gradientsNextLayer(p_PrevLayer->GetLayerSize(), 0);

		for (int i = 0; i < gradientsNextLayer.size(); ++i)
		{
			for (int j = 0; j < gradients.size(); ++j)
			{
				gradientsNextLayer[i] += gradients[j] * m_ActivationPrimeFunc(0.0) * m_Weights[i][j];
			}
		}

		if (!p_PrevLayer)
		{
			// Call another layer, this is going to be hidden
			p_PrevLayer->BackwardPass(gradientsNextLayer);
		}
	}

	/*
	* @derivedValues: derivation computed in the layer above
	*/
	void BackwardPass(const std::vector<double>& derivedValues)
	{

		std::vector<double> gradientsNextLayer(p_PrevLayer->GetLayerSize(), 0);

		for (int i = 0; i < gradientsNextLayer.size(); ++i)
		{
			for (int j = 0; j < derivedValues.size(); ++j)
			{
				gradientsNextLayer[i] += derivedValues[j] * m_ActivationPrimeFunc(0.0) * m_Weights[i][j];
			}
		}

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
		m_Outputs.resize(batchSize, std::vector<double>(m_LayerSize));
		m_Potentials.resize(batchSize, std::vector<double>(m_LayerSize));
	}

	inline const Matrix& GetOutputs() { return m_Outputs; }

	inline size_t GetLayerSize() { return m_LayerSize; }

private:
	void Log()
	{
		std::cout << "Initialized Layer:\n";
		std::cout << "\tInput size : " << m_InputSize << "\n";
		std::cout << "\tLayer size: " << m_LayerSize << "\n";
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
	//
	Layer* p_NextLayer = nullptr;
	Layer* p_PrevLayer = nullptr;
};
