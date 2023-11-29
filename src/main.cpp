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


#define EPOCH_SIZE 20
#define BATCH_SIZE 100
#define TRAINING_SIZE 60'000 // 59'968 //59'904
#define VALIDATION_SIZE 6000 //5888
#define NORMALIZE_DATA 0

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
			double value = std::stod(cell);
			// Data has to be at least scaled to [0.0, 1.0]
			row.push_back(NORMALIZE_DATA ? value : value / 255.0);
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


void ShuffleIndices(std::vector<int>& v)
{
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(v.begin(), v.end(), g);
}

// Returns indices vector, with this indices batches are created -- [0, 25, 1, ...] -> Batch [ [input[0], input[25], input[1]], [...] ] 
std::vector<int> SGDInit(size_t size)
{
	std::vector<int> vec(size);
	std::iota(vec.begin(), vec.end(), 0);
	ShuffleIndices(vec);
	return vec;
}

int main()
{
	std::cout << "Neural network project\n";

	
	std::vector< std::vector<double> > trainData;
	std::vector<int> trainLabels;

	// Data
	LoadMnistData(trainData, "fashion_mnist_train_vectors.csv");
	LoadMnistDataLabels(trainLabels, "fashion_mnist_train_labels.csv");

	if (NORMALIZE_DATA)
	{
		NormalizeData(trainData);
	}

	std::vector<int> sgdTrain = SGDInit(trainData.size());
	std::vector<int> sgdValidate;

	for (int i = TRAINING_SIZE - 1; i >= TRAINING_SIZE - VALIDATION_SIZE; --i)
	{
		sgdValidate.push_back(sgdTrain[i]);
		sgdTrain.pop_back();
	}

	// Checks
	assert(sgdTrain.size() + sgdValidate.size() == TRAINING_SIZE);
	assert(trainData.size() == trainLabels.size() && "Incorrect dims");
	if (trainData.size() % BATCH_SIZE != 0)
	{
		throw std::exception("Batches cannot be created!\n");
	}

	// Net and training

	NeuralNet net({
	Layer(784, 256, ReLu, ReLuPrime),
	Layer(256, 10, Softmax, DoNothing)
		}, BATCH_SIZE);

	for (int epoch = 0; epoch < EPOCH_SIZE; ++epoch)
	{
		std::cout << "Epoch " << epoch+1 << "\n";

		// Training
		for (int j = 0; j < trainData.size() - VALIDATION_SIZE - BATCH_SIZE; j += BATCH_SIZE)
		{
			// Prepare batch
			std::vector<std::vector<double>> trainingData{};
			std::vector <std::vector<int>> trainingLabels{};
			for (int i = 0; i < BATCH_SIZE; ++i)
			{
				trainingData.push_back(trainData[sgdTrain[i + j]]);
				trainingLabels.push_back({ trainLabels[sgdTrain[i + j]] });
			}

			net.Train(trainingData, trainingLabels);
		}

		// Validation
		std::vector<std::vector<double>> trainingData{};
		std::vector<int> trainingLabels{};
		for (int i = 0; i < sgdValidate.size(); ++i)
		{
			trainingData.push_back(trainData[sgdValidate[i]]);
			trainingLabels.push_back(trainLabels[sgdValidate[i]]);
		}

		net.Eval(trainingData, trainingLabels);
		ShuffleIndices(sgdTrain);
	}

	// Test data

	std::cout << "Done!\n";
	return 0;
}