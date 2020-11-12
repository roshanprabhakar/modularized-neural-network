package org.roshanp.NeuralNetwork.Activations;

public class Sigmoid extends Activator {

    @Override
    public double activate(double input) {
        return sigmoid(input);
    }

    @Override
    public double activationDerivative(double input) {
        return sigmoid(input) * (1 - sigmoid(input));
    }

    @Override
    public String representation(double input) {
        return "(1/(1+e^(-1*(" + input + "))))";
    }

    public static double sigmoid(double input) { //capped to prevent overflow error, less exponential calculations
        return -(1 / (1 + Math.exp(input))) + 1;
    }
}
