#include "Activations.h"
#include "NeuralNet.h"
#include <iostream>


int main()
{
	std::cout << "Neural network project\n";
	NeuralNet net({
		Layer(2, 3, Identity),
		Layer(3, 2, Identity)
	});
	
	return 0;
}
