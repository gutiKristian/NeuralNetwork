#include "Activations.h"
#include "NeuralNet.h"
#include <iostream>


int main()
{
	std::cout << "Neural network project\n";
	NeuralNet net({
		Layer(2, 3, Identity, "Id 1"),
		Layer(3, 1, Identity, "Id 2")
	});
	
	return 0;
}