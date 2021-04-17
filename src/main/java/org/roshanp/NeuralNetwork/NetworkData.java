package org.roshanp.NeuralNetwork;

public class NetworkData {

    private Vector input;
    private Vector output;

    public NetworkData(Vector input, Vector correct) {
        this.input = input;
        this.output = correct;
    }

    public Vector getInput() {
        return input;
    }

    public void setInput(Vector input) {
        this.input = input;
    }

    public Vector getOutput() {
        return output;
    }

    public void setOutput(Vector output) {
        this.output = output;
    }

    public static double round(double n, int place) {
        return (int) (n * place) / (double) place;
    }
}
