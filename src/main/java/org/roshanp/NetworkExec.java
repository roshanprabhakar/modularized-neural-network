package org.roshanp;


import org.roshanp.Data.IrisData;
import org.roshanp.NeuralNetwork.Activations.Sigmoid;
import org.roshanp.NeuralNetwork.NetworkData;
import org.roshanp.NeuralNetwork.NeuralNetwork;
import org.roshanp.NeuralNetwork.Vector;

import java.util.ArrayList;
import java.util.Collections;

public class NetworkExec {

    public static void main(String[] args) {

        double trainingPercentage = 0.7;

        ArrayList<NetworkData> data = IrisData.loadIrisData("./src/main/java/org/roshanp/Data/IrisData.csv");
        NetworkData.normalize(data);
        Collections.shuffle(data);

        NeuralNetwork network = new NeuralNetwork(new int[]{2, 2, 1}, 4, 0.0001, 10, 0.001, NeuralNetwork.sigmoid);
        network.train(data, true);

        for (int i = 0; i < data.size(); i++) {
            System.out.println("----------------------");
            System.out.println("Test input: " + data.get(i).getInput());
            System.out.println("Test actual: " + data.get(i).getOutput());
            NeuralNetwork.ForwardPropOutput output = network.forwardProp(data.get(i).getInput());
            System.out.println("Network output: " + output.getResultant());
            System.out.println("----------------------");
        }

        System.out.println("cumulative loss: " + NeuralNetwork.cumulativeLoss(data, network));
    }
}

