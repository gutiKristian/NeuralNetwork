#include "Activations.h"
#include "NeuralNet.h"
#include <iostream>
#include <vector>
#include <random>
#include <ctime>

int main()
{
	std::cout << "Neural network project\n";

	std::vector< std::vector< std::vector<double> > > batches
	{
		{{1, 0}}, {{0, 1}}, {{1, 1}}, {{0, 0}}
	};

	std::srand(std::time(0));

	int batchSize = 1;


	// generate this on the fly
	std::vector<std::vector<std::vector<double>>> batchesOuts
	{
		{{1}}, {{1}}, {{0}}, {{0}}
	};
	

	NeuralNet net({
	Layer(2, 2, ReLu, ReLuPrime),
	Layer(2, 1, LogisticSigmoid, LogisticSigmoidPrime)
		}, batchSize);

	for (int epoch = 0; epoch < 10000; ++epoch)
	{
		std::cout << "Epoch " << epoch << "\n";
		for (int i = 0; i < batches.size(); ++i)
		{
			net.Train(batches[i], batchesOuts[i]);
			std::cout << "Error: " << net.ComputeError(batchesOuts[i]) << "\n";
		}
	}


	net.Eval({ {1, 0}, {0, 0}, {0, 1}, {1, 1} }, { {1}, {0}, {1}, {0} });

	std::cout << "Done!\n";
	return 0;
}