package org.roshanp.NeuralNetwork;

import java.util.ArrayList;

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

    public static void normalize(ArrayList<NetworkData> data) {
        Vector mean = mean(data);
        Vector stdDev = stdDev(data);
        for (NetworkData d : data) {
            for (int i = 0; i < d.getInput().length(); i++) {
                d.getInput().set(i, (d.getInput().get(i) - mean.get(i)) / stdDev.get(i));
            }
        }
    }

    // only considers data input
    public static Vector mean(ArrayList<NetworkData> data) {
        Vector out = new Vector(data.get(0).getInput().length());
        for (NetworkData d : data) {
            out.add(d.getInput());
        }
        out.multiplyScalar(1.0 / data.size());
        return out;
    }

    //only considers data input
    public static Vector stdDev(ArrayList<NetworkData> data) {
        Vector mean = mean(data);
        Vector sd = new Vector(data.get(0).getInput().length());
        for (NetworkData d : data) {
            sd.add(mean.copy().add(d.getInput().copy().multiplyScalar(-1)).square());
        }
        return sd.multiplyScalar(1.0 / data.size()).raise(0.5);
    }
}
