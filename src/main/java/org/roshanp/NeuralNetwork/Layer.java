package org.roshanp.NeuralNetwork;

import org.roshanp.NeuralNetwork.Activations.Activator;
import org.roshanp.NeuralNetwork.Activations.Sigmoid;

public class Layer implements NetworkConstants {

    private Perceptron[] neurons;

    public Layer(int numberOfPerceptrons, int dataLength, Activator activator) {
        neurons = new Perceptron[numberOfPerceptrons];
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Perceptron(activator, dataLength, true);
        }
    }

    public Perceptron get(int i) {
        return neurons[i];
    }

    public Vector activations(Vector previous) { //activations from previous layer
        Vector activations = new Vector(neurons.length);
        for (int i = 0; i < neurons.length; i++) {
            activations.set(i, neurons[i].guess(previous));
        }
        return activations;
    }

    public int getInputLength(int neuron) {
        return neurons[neuron].getWeights().length();
    }

    public int length() {
        return neurons.length;
    }

    public String toString() {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < neurons.length; i++) {
            out.append("[").append(neurons[i].getWeights()).append("]{").append(NetworkData.round(neurons[i].getBias(), 100)).append("}").append("   ");
        }
        return out.toString();
    }
}
