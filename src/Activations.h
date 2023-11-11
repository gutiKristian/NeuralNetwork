#pragma once

#include <cmath>
#include <algorithm>

using ActivationFunction = double (*)(double);

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

// hidden layer
double ReLu(double potential)
{
	return std::max(potential, 0.0);
}

double Identity(double potential)
{
	return potential;
}