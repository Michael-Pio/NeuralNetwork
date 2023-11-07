#include "NeuralNetwork.h"

int main() {
	NeuralNetwork NN1 = NeuralNetwork(std::vector<unsigned int>{ 17, 20, 15,2 });
	NeuralNetwork NN2 = NeuralNetwork(std::vector<unsigned int>{ 17, 20, 15,2 });


	NN1.saveModel("Models\\P1.csv");
	NN2.saveModel("Models\\P2.csv");

	NeuralNetwork NN3 = NeuralNetwork::CrossOver(NN1, NN2,0.2);

	NN3.saveModel("Models\\O1.csv");
	//TO DO Implement Training Methods

	NN3.Mutate(0.5,0.2);

	NeuralNetwork NN4 = NN3;
	NN4.saveModel("Models\\OO1.csv");
	std::vector<double> inpu {1, 2, 3, 4, 6, 7, 6, 5, 4, 2, 2, 4, 6, 7, 6, 43, 3};
	NN4.feedForward(inpu);

	NN1.displayDetails();
	NN2.displayDetails();
	NN3.displayDetails();
}
