#include "Activations.h"
#include "NeuralNet.h"
#include <iostream>


int main()
{
	std::cout << "Neural network project\n";
	auto topLoss = [](int x) -> bool { return (x == 0); };

	NeuralNet net({
		Layer(2, 3, Identity, topLoss), // add ptr to 
		Layer(3, 2, Identity, topLoss)
	});
	
	return 0;
}
