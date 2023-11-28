#include "Activations.h"
#include "NeuralNet.h"
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <sstream>


#define EPOCH_SIZE 100
#define BATCH_SIZE 16
#define TRAINING_SIZE 60'000
#define VALIDATION_SIZE 6000


/*
* Normalizes data and then scales them into [0, 1] interval.
*/
void NormalizeData(std::vector< std::vector<double> >& data)
{
	std::cout << "Normalizing...";
	double mean = 0.0;
	for (auto& img : data)
	{
		mean += std::accumulate(img.begin(), img.end(), 0.0);
	}
	mean /= static_cast<double>(data.size());

	double stddev = 0.0;
	for (auto& img : data)
	{
		for (const auto& value : img)
		{
			stddev += std::pow(value - mean, 2);
		}
	}

	stddev = std::sqrt(stddev / static_cast<double>(data.size()));
	
	for (auto& img : data)
	{
		for (auto& value : img)
		{
			value = (value - mean) / stddev;
		}
	}

	for (auto& img : data)
	{
		double minElement = *std::min_element(img.begin(), img.end());
		double maxElement = *std::max_element(img.begin(), img.end());
		for (auto& value : img)
		{
			value = (value - minElement) / (maxElement - minElement);
		}

	}
	std::cout << "Done\n";
}


void LoadMnistData(std::vector< std::vector<double>>& data, std::string name)
{
	std::string path = "../../../data/" + name;
	std::cout << "Loading: " << path << "...";
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
			row.push_back(std::stod(cell));
		}

		data.push_back(row);
		if (data.size() == TRAINING_SIZE)
		{
			break;
		}
	}

	// Close the file
	file.close();
	std::cout << "Done\n";
}

void LoadMnistDataLabels(std::vector<int>& labels, std::string name)
{
	std::string path = "../../../data/" + name;
	std::cout << "Loading: " << path << "...";
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
	std::cout << "Done\n";
}


int main()
{
	std::cout << "Neural network project\n";

	
	std::vector< std::vector<double> > trainData;
	std::vector<int> trainLabels;

	LoadMnistData(trainData, "fashion_mnist_train_vectors.csv");
	LoadMnistDataLabels(trainLabels, "fashion_mnist_train_labels.csv");

	NormalizeData(trainData);


	NeuralNet net({
	Layer(784, 64, ReLu, ReLuPrime),
	Layer(64, 10, Softmax, DoNothing)
		}, BATCH_SIZE);

	for (int epoch = 0; epoch < EPOCH_SIZE; ++epoch)
	{
		std::cout << "Epoch " << epoch+1 << "\n";

		if (epoch > 2)
		{
			net.AdjustLr(0.0001);
		}
		
		for (int j = 0; j < trainData.size(); j += BATCH_SIZE)
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