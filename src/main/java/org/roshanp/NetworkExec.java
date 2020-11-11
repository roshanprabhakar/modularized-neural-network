package org.roshanp;


import org.roshanp.NeuralNetwork.NetworkData;
import org.roshanp.NeuralNetwork.NeuralNetwork;
import org.roshanp.NeuralNetwork.Vector;

import java.util.ArrayList;

public class NetworkExec {

    public static void main(String[] args) {

        NeuralNetwork network = new NeuralNetwork(new int[]{2, 2, 1}, 3, 0.001, 1, 0.001, 0.001);

        ArrayList<NetworkData> trainingData = new ArrayList<>() {
            {
                add(new NetworkData(new Vector(3, true), new Vector(1, true)));
                add(new NetworkData(new Vector(3, true), new Vector(1, true)));
                add(new NetworkData(new Vector(3, true), new Vector(1, true)));
            }
        };

        network.train(trainingData);
    }
}

