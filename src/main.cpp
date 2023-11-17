#include "Activations.h"
#include "NeuralNet.h"
#include <iostream>


int main()
{
	std::cout << "Neural network project\n";

	NeuralNet net({
		Layer(2, 3, Identity, topLoss), // add ptr to 
		Layer(3, 2, Identity, topLoss)
	});
	
	return 0;
}
