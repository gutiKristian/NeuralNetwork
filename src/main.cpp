#include "Activations.h"
#include "NeuralNet.h"
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>


#define EPOCH_SIZE 100
#define BATCH_SIZE 1
#define TRAINING_SIZE 10'000
#define VALIDATION_SIZE 1000

void LoadMnistData(std::vector< std::vector<double>>& data, std::string name)
{
	std::string path = "../../../data/" + name;
	std::ifstream file(path);

	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << path << std::endl;
	}

	std::string line;

	// Read each line from the CSV file
	while (std::getline(file, line))
	{
		std::vector<double> row;
		std::stringstream ss(line);
		std::string cell;

		while (std::getline(ss, cell, ','))
		{
			row.push_back((std::stod(cell) - 127.5) / 127.5);
		}

		data.push_back(row);
		if (data.size() == TRAINING_SIZE)
		{
			break;
		}
	}

	// Close the file
	file.close();
}


void LoadMnistDataLabels(std::vector<int>& labels, std::string name)
{
	std::string path = "../../../data/" + name;
	std::ifstream file(path);

	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << path << std::endl;
	}

	std::string line;

	// Read each line from the CSV file
	while (std::getline(file, line))
	{
		// Convert the line to an integer and add it to the data vector
		labels.push_back(std::stoi(line));

		if (labels.size() == TRAINING_SIZE)
		{
			break;
		}
	}	

	// Close the file
	file.close();
}


int main()
{
	std::cout << "Neural network project\n";

	
	std::vector< std::vector<double> > trainData;
	std::vector<int> trainLabels;

	LoadMnistData(trainData, "fashion_mnist_train_vectors.csv");
	LoadMnistDataLabels(trainLabels, "fashion_mnist_train_labels.csv");



	NeuralNet net({
	Layer(784, 256, ReLu, ReLuPrime),
	Layer(256, 10, Softmax, DoNothing)
		}, BATCH_SIZE);

	for (int epoch = 0; epoch < EPOCH_SIZE; ++epoch)
	{
		std::cout << "Epoch " << epoch+1 << "\n";
		
		for (int j = 0; j < trainData.size(); ++j)
		{
			std::vector<std::vector<double>> trainingData{};
			std::vector <std::vector<int>> trainingLabels{};
			for (int i = 0; i < BATCH_SIZE; ++i)
			{
				trainingData.push_back(trainData[i + j]);
				trainingLabels.push_back({ trainLabels[i + j] });
			}

			net.Train(trainingData, trainingLabels);
		}

		std::vector<std::vector<double>> trainingData{};
		std::vector<int> trainingLabels{};
		for (int i = trainData.size() - VALIDATION_SIZE; i < trainData.size(); ++i)
		{
			trainingData.push_back(trainData[i]);
			trainingLabels.push_back(trainLabels[i]);
		}
		net.Eval(trainingData, trainingLabels);

	}

	std::cout << "Done!\n";
	return 0;
}