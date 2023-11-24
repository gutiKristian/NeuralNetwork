#include "Activations.h"
#include "NeuralNet.h"
#include <iostream>
#include <vector>

int main()
{
	std::cout << "Neural network project\n";
	
	std::vector<std::vector<double>> batch{ {1, 1}, {10, 2}, {13, 5}, {14, 8},{17, 23} };
	std::vector<std::vector<double>> gt{ {2}, {12}, {18}, {22}, {40} };
	
	NeuralNet net({
	Layer(2, 2, Identity, Identity), // add ptr to 
	Layer(2, 2, Identity, Identity)
	}, batch.size());

	net.Train(batch, gt);
	std::cout << "Done!\n";
	return 0;
}
