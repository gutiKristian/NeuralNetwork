#pragma once

#include <cmath>
#include <algorithm>

using ActivationFunction = double (*)(double);

double UnitStepFunction(double potential);

// output layer
double LogisticSigmoid(double potential);

// hidden layer
double ReLu(double potential);

double ReLuPrime(double potential);

double Identity(double potential);

double IdentityPrime(double potential);