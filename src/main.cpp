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
#include <chrono>


#define EPOCH_SIZE 100
#define BATCH_SIZE 100
#define TRAINING_SIZE 60'000 // 59'968 //59'904
#define VALIDATION_SIZE 12000 //5888
#define NORMALIZE_DATA 0
#define STOPPING_ACC 90.85
#define RESHUFFLE_ALL 1

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

/*
* Fills arrays with indices. tI - training indices, vI - validation Indices
*/
void PrepareIndices(const std::vector<int>& shuffledIndices, std::vector<int>& tI, std::vector<int>& vI)
{
	assert(tI.size() + vI.size() == TRAINING_SIZE);

	// Fill in validation
	for (size_t i = 0; i < VALIDATION_SIZE; ++i)
	{
		vI[i] = shuffledIndices[i];
	}

	// Fill in training
	for (size_t i = VALIDATION_SIZE; i < TRAINING_SIZE; ++i)
	{
		tI[i - VALIDATION_SIZE] = shuffledIndices[i];
	}

}

int main()
{

	static_assert(((TRAINING_SIZE - VALIDATION_SIZE) % BATCH_SIZE == 0) && "Batches cannot be created!\n");


	std::cout << "Neural network project\n";

	auto timeStart = std::chrono::high_resolution_clock::now();
	
	std::vector< std::vector<double> > trainData;
	std::vector<int> trainLabels;

	// Data
	LoadMnistData(trainData, "fashion_mnist_train_vectors.csv");
	LoadMnistDataLabels(trainLabels, "fashion_mnist_train_labels.csv"); 

	std::vector< std::vector<double> > testData;
	std::vector<int> testLabels;

	LoadMnistData(testData, "fashion_mnist_test_vectors.csv");
	LoadMnistDataLabels(testLabels, "fashion_mnist_test_labels.csv");

	assert(trainData.size() == TRAINING_SIZE);
	assert(trainLabels.size() == TRAINING_SIZE);

	if (NORMALIZE_DATA)
	{
		NormalizeData(trainData);
		NormalizeData(testData);
	}

	std::cout << "Data loaded after: " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - timeStart) << " seconds\n";
	//////////////////////////////////////////////////////////////////////////
	// 
	//						PREPARE INDICES

	std::random_device rd;
	std::mt19937 generator(1337);

	// Fill with [0,1,2,...,trainData.size()]
	std::vector<int> indices(trainData.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::shuffle(indices.begin(), indices.end(), generator);


	std::vector<int> sgdTrain;
	sgdTrain.resize(TRAINING_SIZE - VALIDATION_SIZE, 0);
	std::vector<int> sgdValidate;
	sgdValidate.resize(VALIDATION_SIZE, 0);

	PrepareIndices(indices, sgdTrain, sgdValidate);

	//////////////////////////////////////////////////////////////////////////
	//						NET TRAINING

	NeuralNet net({
	Layer(784, 64, ReLu, ReLuPrime),
	Layer(64, 32, ReLu, ReLuPrime),
	Layer(32, 10, Softmax, DoNothing)
		}, BATCH_SIZE);

	std::cout << "\n\n";
	for (int epoch = 0; epoch < EPOCH_SIZE; ++epoch)
	{
		std::cout << "\t\tEpoch " << epoch+1 << "\n";

		// Training

		// Validation
		std::vector<std::vector<double>> validationData{};
		std::vector<int> validationLabels{};
		for (size_t i = 0; i < VALIDATION_SIZE; ++i)
		{
			validationData.push_back(trainData[sgdValidate[i]]);
			validationLabels.push_back(trainLabels[sgdValidate[i]]);
		}


		constexpr int SGD_TRAIN_SIZE = TRAINING_SIZE - VALIDATION_SIZE;

		// Batches
		for (int j = 0; j < SGD_TRAIN_SIZE - BATCH_SIZE; j += BATCH_SIZE)
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

		double acc = net.Eval(validationData, validationLabels);
		/*	if (acc > 89.0)
			{
				net.AdjustLr(0.001);
			}*/

		if (acc > STOPPING_ACC)
		{
			std::cout << "\nEnding..\n";
			break;
		}

		// Reshuffle
		if (RESHUFFLE_ALL)
		{
			// We shuffle whole train data and also change the validation set
			std::shuffle(indices.begin(), indices.end(), generator);
			PrepareIndices(indices, sgdTrain, sgdValidate);
		}
		else
		{
			// Just change batches, validation stays the same for entire training
			std::shuffle(sgdTrain.begin(), sgdTrain.end(), generator);
		}

		

		std::cout << "Epoch " << epoch + 1 << " done, time elapsed:\n";
		auto timeEpochDone = std::chrono::high_resolution_clock::now();
		auto timeDurationSeconds = std::chrono::duration_cast<std::chrono::seconds>(timeEpochDone - timeStart);
		auto timeDurationMinutes = std::chrono::duration_cast<std::chrono::minutes>(timeDurationSeconds);
		std::cout << timeDurationMinutes.count() << " minute(s), " << (timeDurationSeconds - timeDurationMinutes).count() << " seconds\n";
		std::cout << "\n";
	}

	// Test data
	std::cout << "Evaluating test data...\n";
	net.Eval(testData, testLabels);

	std::cout << "Done!\n";
	return 0;
}