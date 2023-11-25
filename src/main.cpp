#include "Activations.h"
#include "NeuralNet.h"
#include <iostream>
#include <vector>
#include <random>
#include <ctime>

int main()
{
	std::cout << "Neural network project\n";
	
	std::vector< std::vector< std::vector<double> > > batches{};

	std::srand(std::time(0));

	int batchSize = 1;

	for (int j = 0; j < 500; ++j)
	{
		batches.push_back({});
		for (int i = 0; i < batchSize; ++i)
		{
			int randomInt1 = std::rand() % 500 + 1; // Generates a random number between 1 and 100
			int randomInt2 = std::rand() % 500 + 1; // Generates a random number between 1 and 100
			batches[j].push_back({ static_cast<double>(randomInt1 / 500.0), static_cast<double>(randomInt2 / 500.0) });
		}
	}
	

	// generate this on the fly
	std::vector<std::vector<double>> batchesOuts{};
	batchesOuts.resize(batchSize, std::vector<double>(1, 0.0));

	NeuralNet net({
	Layer(2, 2, ReLu, ReLuPrime), // add ptr to 
	Layer(2, 1, Identity, IdentityPrime)
	}, batchSize);

	for (int epoch = 0; epoch < 1000; ++epoch)
	{
		std::cout << "Epoch " << epoch << "\n";
		for (const auto& batch : batches)
		{
			for (int k = 0; k < batchSize; ++k)
			{
				batchesOuts[k][0] = batch[k][0] + batch[k][1];
			}

			net.Train(batch, batchesOuts);
		}
		std::cout << "Error: " << net.ComputeError(batchesOuts) << "\n";
	}

	std::cout << "Done!\n";
	return 0;
}
