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
		{{1, 0}, {0, 1}, {1, 1}, {0, 0}}
	};

	int batchSize = 4;

	std::vector<std::vector<std::vector<double>>> batchesOuts
	{
		{{1}, {1}, {0}, {0}}
	};
	

	NeuralNet net({
	Layer(2, 5, ReLu, ReLuPrime),
	Layer(5, 1, Softmax, Softmax)
		}, batchSize);

	for (int epoch = 0; epoch < 20000; ++epoch)
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