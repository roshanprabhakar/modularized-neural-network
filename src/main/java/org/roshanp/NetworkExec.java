package org.roshanp;


import org.roshanp.Data.IrisData;
import org.roshanp.NeuralNetwork.Activations.Sigmoid;
import org.roshanp.NeuralNetwork.NetworkData;
import org.roshanp.NeuralNetwork.NetworkVisualizer;
import org.roshanp.NeuralNetwork.NeuralNetwork;
import org.roshanp.NeuralNetwork.Vector;

import java.util.ArrayList;
import java.util.Collections;

public class NetworkExec {

    public static void main(String[] args) throws InterruptedException {

        ArrayList<NetworkData> data = IrisData.loadIrisData("./src/main/java/org/roshanp/Data/IrisData.csv");
        NetworkData.normalize(data);
        Collections.shuffle(data);
        
//        ArrayList<NetworkData> trainingData = new ArrayList<>();
//        for (int i = 0; i < data.size() * trainingPercentage; i++) {
//            trainingData.add(data.get(i));
//        }
//
//        ArrayList<NetworkData> testData = new ArrayList<>();
//        for (int i = (int) (data.size() * trainingPercentage); i < data.size(); i++) {
//            testData.add(data.get(i));
//        }

        NeuralNetwork network = new NeuralNetwork(new int[]{4,3}, 4, 0.01, 10, 0.001, NeuralNetwork.sigmoid);
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

