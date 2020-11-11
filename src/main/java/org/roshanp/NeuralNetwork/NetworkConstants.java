package org.roshanp.NeuralNetwork;

public interface NetworkConstants {

    double LEARNING_RATE = 1;
    int EPOCHS = 3000;
//    int POWER = 3; //powers > 1 passable only to single l&n perceptrons
    int POWER = 1;

    String TARGET = "Iris-setosa";
    //private static final String TARGET = "target";
    int DATA_LENGTH = 2;

    double SEPARABLE = 0.95;

    //For visualization purposes
    int NUM_DATA_GROUPS = 3;
}
