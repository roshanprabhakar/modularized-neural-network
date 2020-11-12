package org.roshanp.NeuralNetwork;

import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.roshanp.NeuralNetwork.Activations.Activator;

import java.util.ArrayList;
import java.util.Collections;

public class Perceptron {

    private final Vector weights;
    private double bias;

    private final Activator activator;

    public Perceptron(Activator activator, int dataLength, boolean randomInitializations) {
        this.activator = activator;
        if (randomInitializations) {
            weights = new Vector(dataLength, true); //don't want to replace vector initializations everywhere, just created a second constructor
            bias = Math.random();
        } else {
            weights = new Vector(dataLength);
            bias = 0;
        }
    }

    public double guess(Vector input) {
        return activator.activate(weights.dotProduct(input) + bias);
    }

    public double unactivatedGuess(Vector input) {
        return weights.dotProduct(input) + bias;
    }

    public Vector getWeights() {
        return this.weights;
    }
    public double getBias() {
        return bias;
    }

    public void shiftWeight(int weight, double shift) {
        weights.set(weight, weights.get(weight) + shift);
    }
    public void updateWeight(int weight, double newWeight) {
        weights.set(weight, newWeight);
    }
    public void updateBias(double newBias) {
        this.bias = newBias;
    }

    public Activator getActivator() {
        return activator;
    }
}
