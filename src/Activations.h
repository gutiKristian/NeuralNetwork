#pragma once

#include <cmath>
#include <algorithm>
#include <vector>

//! Potential input is actually an output array to be modified
using ActivationFunction = void (*)(std::vector<double>&);

void UnitStepFunction(std::vector<double>& potentials);

void LogisticSigmoid(std::vector<double>& potentials);

void LogisticSigmoidPrime(std::vector<double>& potentials);

void ReLu(std::vector<double>& potentials);

void ReLuPrime(std::vector<double>& potentials);

void Identity(std::vector<double>& potentials);

void IdentityPrime(std::vector<double>& potentials);

void Softmax(std::vector<double>& potentials);

void DoNothing(std::vector<double>& potentials);