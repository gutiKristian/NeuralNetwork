#include "Activations.h"
#include <cmath>

void UnitStepFunction(std::vector<double>& potentials)
{
	auto size = potentials.size();
	for (int i = 0; i < size; ++i)
	{
		potentials[i] = potentials[i] >= 0.0 ? 1.0 : 0.0;
	}
}

// output layer
void LogisticSigmoid(std::vector<double>& potentials)
{
	auto size = potentials.size();
	double steepness = 1.0;
	for (int i = 0; i < size; ++i)
	{
		potentials[i] = 1.0 / (1.0 + std::exp(-potentials[i] * steepness));
	}
}

void LogisticSigmoidPrime(std::vector<double>& potentials)
{
	auto size = potentials.size();
	double steepness = 1.0;
	for (int i = 0; i < size; ++i)
	{
		double l = 1.0 / (1.0 + std::exp(-potentials[i] * steepness));
		potentials[i] = l * (1.0 - l);
	}
}

// hidden layer
void ReLu(std::vector<double>& potentials)
{
	auto size = potentials.size();
	for (int i = 0; i < size; ++i)
	{
		potentials[i] = std::max(potentials[i], 0.0);
	}
}

void ReLuPrime(std::vector<double>& potentials)
{
	auto size = potentials.size();
	for (int i = 0; i < size; ++i)
	{
		potentials[i] = potentials[i] > 0.0 ? 1.0 : 0.0;
	}
}

void Identity(std::vector<double>& potentials)
{
}

void IdentityPrime(std::vector<double>& potentials)
{
	auto size = potentials.size();
	for (int i = 0; i < size; ++i)
	{
		potentials[i] = 1.0;
	}
}

void Softmax(std::vector<double>& potentials)
{
	double _sum = 0.0;
	for (auto& val : potentials)
	{
		val = std::exp(val);
		_sum += val;
	}

	for (auto& val : potentials)
	{
		val /= _sum;
	}
}

void DoNothing(std::vector<double>& potentials)
{
}
