package org.roshanp.NeuralNetwork.Activations;

public abstract class Activator {

    public abstract double activate(double input);
    public abstract double activationDerivative(double input);
    public abstract String representation(double input);
}
