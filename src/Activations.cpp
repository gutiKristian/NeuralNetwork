#include "Activations.h"
#include <cmath>

double UnitStepFunction(double potential)
{
	return potential >= 0.0 ? 1.0 : 0.0;
}

// output layer
double LogisticSigmoid(double potential)
{
	double steepness = 1.0;
	return 1 / (1 + std::exp(potential * steepness));
}

double LogisticSigmoidPrime(double potential)
{
	double l = LogisticSigmoid(potential);
	return l * (1 - l);
}

// hidden layer
double ReLu(double potential)
{
	return std::max(potential, 0.0);
}

double ReLuPrime(double potential)
{
	return potential > 0.0 ? 1.0 : 0.0;
}

double Identity(double potential)
{
	return potential;
}

double IdentityPrime(double potential)
{
	return 1.0;
}

double Tanh(double potential)
{
	return std::tanh(potential);
}

double TanhPrime(double potential)
{
	return 1 - std::pow(std::tanh(potential), 2);
}
