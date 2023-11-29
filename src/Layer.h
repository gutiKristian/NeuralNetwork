#pragma once
#include "Activations.h"
#include <cassert>
#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <algorithm>

class Layer
{
	using Matrix = std::vector<std::vector<double>>;
public:
	Layer(size_t In, size_t Size, ActivationFunction activationFunction, ActivationFunction activationPrime) : 
		m_InputSize(In), m_LayerSize(Size), m_ActivationFunc(activationFunction), m_ActivationPrimeFunc(activationPrime)
	{
		m_Weights.resize(m_LayerSize, std::vector<double>(m_InputSize));
		m_Momentum.resize(m_LayerSize, std::vector<double>(m_InputSize, 0.0));
		m_Bias.resize(m_LayerSize, 0.0);
		InitWeights();
	};

public:

	std::vector<double> AverageArray(const Matrix& arr)
	{
		assert(arr.size() > 0);

		std::vector<double> out{};
		out.resize(arr[0].size(), 0.0);
		auto arrSize = arr[0].size();

		for (int k = 0; k < arr.size(); ++k)
		{
			for (int i = 0; i < arrSize; ++i)
			{
				out[i] += arr[k][i];
			}
		}

		for (auto& elem : out)
		{
			elem /= arrSize;
		}

		return out;
	}

	/*
	* Input is 2D array of inputs -- batch
	*/
	void Forward(const Matrix& batch)
	{
		assert(batch.size() > 0 && "Batch size must be > 0");
		assert(m_InputSize == batch[0].size()); // assume its uniform 2D array

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
				m_Outputs[k][i] = current; // set outputs to potential as we will pass to activation func
				m_PrimeOutputs[k][i] = current; // this value is finally computed during backprop. with prime activation function
			
			}
			
			m_ActivationFunc(m_Outputs[k]); // call activation
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
	void Backward(const std::vector< std::vector<int> >& trueValuesBatch)
	{
		assert(trueValuesBatch.size() == m_Outputs.size() && m_Outputs.size() > 0 && "Size mismatch!");

		auto batchSize = m_Outputs.size();
		auto batchOutputSize = m_Outputs[0].size();

		Matrix gradients;
		gradients.resize(batchSize, std::vector<double>(batchOutputSize, 0.0));

		// Compute gradient for each batch
		for (int k = 0; k < batchSize; ++k)
		{
			for (int i = 0; i < batchOutputSize; ++i)
			{
				gradients[k][i] = ((i == trueValuesBatch[k][0]) ? m_Outputs[k][i] - 1.0 : m_Outputs[k][i]);
			}
		}

		/*
		* Update weights and biases below output layer.
		*/

		auto nextLayerSize = p_PrevLayer->GetLayerSize();
		const Matrix& y_i = p_PrevLayer->GetOutputs();

		for (int i = 0; i < m_LayerSize; ++i)
		{
			for (int j = 0; j < nextLayerSize; ++j)
			{
				double weigthDer = 0.0;
				for (int k = 0; k < batchSize; ++k)
				{
					weigthDer += gradients[k][i] * y_i[k][j];
				}

				weigthDer /= batchSize;
				m_Weights[i][j] += -m_LearningRate * weigthDer + m_Momentum[i][j] * m_MomentumAlpha;
				m_Momentum[i][j] = -m_LearningRate * weigthDer + m_Momentum[i][j] * m_MomentumAlpha;
			}
		}

		// Update biases on this layer
		for (int i = 0; i < m_LayerSize; ++i)
		{
			double biasDer = 0.0;
			for (int k = 0; k < batchSize; ++k)
			{
				biasDer += gradients[k][i];
			}
			biasDer /= batchSize;
			m_Bias[i] += -m_LearningRate * biasDer;
		}

		/*
		* Calculate y_j for next layer. This was derived from the slides.
		* Keep this derivations in batch form as we will average them during the learning phase. (For now ???)
		*/
		Matrix inputNextLayer;
		inputNextLayer.resize(batchSize, std::vector<double>(nextLayerSize, 0.0));

		for (int k = 0; k < batchSize; ++k)
		{
			for (int l = 0; l < nextLayerSize; ++l)
			{
				double y_l = 0.0;
				for (int i = 0; i < m_LayerSize; ++i)
				{
					y_l += gradients[k][i] * m_Weights[i][l];
				}
				inputNextLayer[k][l] = y_l;
			}
		}

		p_PrevLayer->Backward(inputNextLayer);
	}

	/*
	* @derivedValues: derivation computed in the layer above
	*/
	void Backward(const Matrix& inputDerivation)
	{

		if (p_PrevLayer == nullptr)
		{
			return;
		}

		assert(inputDerivation.size() > 0 && "Empty inputDerivation");


		auto batchSize = inputDerivation.size();
		// Next layer in backpropagation is layer that is previous to this one
		auto nextLayerSize = p_PrevLayer->GetLayerSize();


		for (int k = 0; k < batchSize; ++k)
		{
			m_ActivationPrimeFunc(m_PrimeOutputs[k]);
		}

		/*
		* UNCOMMENT IF WANT TO USE MORE THAN ONE HIDDEN LAYER
		* Compute gradients y_j based on the gradients computed above (that were passed here).
		*/
		
		Matrix inputNextLayer; // can be preallocated
		
		inputNextLayer.resize(batchSize, std::vector<double>(nextLayerSize, 0.0));

		for (int k = 0; k < batchSize; ++k)
		{
			for (int j = 0; j < nextLayerSize; ++j)
			{
				double y_j = 0.0;
				for (int r = 0; r < m_LayerSize; ++r)
				{
					y_j += inputDerivation[k][r] * m_PrimeOutputs[k][r] * m_Weights[r][j];
				}
				inputNextLayer[k][j] = y_j;
			}
		}

		/*
		* Update the weights and biases.
		*/

		const Matrix& y_i = p_PrevLayer->GetOutputs();
		for (int i = 0; i < m_LayerSize; ++i)
		{
			for (int j = 0; j < nextLayerSize; ++j)
			{
				double weigthDer = 0.0;
				for (int k = 0; k < batchSize; ++k)
				{
					weigthDer += inputDerivation[k][i] * m_PrimeOutputs[k][i] * y_i[k][j];
				}
				weigthDer /= batchSize;
				m_Weights[i][j] += -m_LearningRate * weigthDer + m_Momentum[i][j] * m_MomentumAlpha;
				m_Momentum[i][j] = -m_LearningRate * weigthDer + m_Momentum[i][j] * m_MomentumAlpha;
			}
		}

		for (int i = 0; i < m_LayerSize; ++i)
		{
			double biasDer = 0.0;
			for (int k = 0; k < batchSize; ++k)
			{
				biasDer += inputDerivation[k][i] * m_PrimeOutputs[k][i];
			}
			biasDer /= batchSize;
			m_Bias[i] += -m_LearningRate * biasDer;
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
		m_PrimeOutputs.resize(batchSize, std::vector<double>(m_LayerSize));
		m_Potentials.resize(batchSize, std::vector<double>(m_LayerSize));
	}

	inline const Matrix& GetOutputs() { return m_Outputs; }

	// We do not want to copy the data, Layer class will never outlive the data
	Matrix& SetOutput() { return m_Outputs; }

	size_t GetLayerSize() { return m_LayerSize; }

	size_t GetInputSize() { return m_InputSize; }

	const Matrix& GetWeights() { return m_Weights; }

	void SetLearningRate(double lr) { m_LearningRate = lr; }

private:
	
	void InitWeights()
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		// Set the mean and standard deviation for the normal distribution
		//double mean = 0.0;
		//double stddev = 0.01;
		//std::normal_distribution<double> distribution(mean, stddev);
		
		std::normal_distribution<double> distribution(0.0f, std::sqrt(2.0 / (m_LayerSize + m_InputSize)));

		// std::normal_distribution<double> distribution(0.0f, 2.0 / (m_LayerSize + m_InputSize));


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
	Matrix m_PrimeOutputs{};
	Matrix m_Momentum{};
	// Backpropagation and learning
	double m_LearningRate = 0.01;
	double m_MomentumAlpha = 0.9;
	//
	Layer* p_NextLayer = nullptr;
	Layer* p_PrevLayer = nullptr;
};
