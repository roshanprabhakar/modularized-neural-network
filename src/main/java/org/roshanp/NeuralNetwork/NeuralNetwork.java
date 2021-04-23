package org.roshanp.NeuralNetwork;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.roshanp.NeuralNetwork.Activations.Activator;
import org.roshanp.NeuralNetwork.Activations.Sigmoid;

import javax.swing.*;
import java.util.ArrayList;

//IMPORTANT
//all data must be bounded in input and output by the scope of the activation function (layers 0 and n are activated by the specified activation, therefore bounded)
public class NeuralNetwork {

    public static final Activator sigmoid = new Sigmoid();

    //primary network object
    private ArrayList<Layer> network;

    //input data dimension
    private int inputSize;

    //learningRate * maxMomentum = largest possible step size
    private double learningRate;
    private double maxMomentum;

    //training ends when gradient momentum reaches or surpasses minMomentum
    private double minMomentum;

    /**
     * @param layers layers[i] corresponds to the number of neurons at layer i in the network, where i is the layer index AFTER input
     * @param inputSize input dimension
     * @param learningRate one of the factors for each component of the gradient
     * @param maxMomentum max bounds for accelerating gradient updates
     * @param minMomentum min bounds for decelerating gradient updates
     */
    public NeuralNetwork(int[] layers, int inputSize, double learningRate, double maxMomentum, double minMomentum, Activator activator) {
        network = new ArrayList<>();
        network.add(new Layer(layers[0], inputSize, activator));
        for (int layer = 1; layer < layers.length; layer++) {
            network.add(new Layer(layers[layer], layers[layer - 1], activator));
        }
        this.inputSize = inputSize;
        this.learningRate = learningRate;
        this.maxMomentum = maxMomentum;
        this.minMomentum = minMomentum;
    }

    /**
     * @param inputSize input dimension
     * @param neuronsPerHidden constant neuron count among all layers
     * @param numHiddenLayers number of layers in the network, including the output layer
     * @param learningRate one of the factors for each component of the gradient
     * @param maxMomentum max bounds for accelerating gradient updates
     * @param minMomentum min bounds for decelerating gradient updates
     */
    public NeuralNetwork(int inputSize, int neuronsPerHidden, int numHiddenLayers, double learningRate, double maxMomentum, double minMomentum, Activator activator) {
        network = new ArrayList<>();
        network.add(new Layer(neuronsPerHidden, inputSize, activator));
        for (int i = 0; i < numHiddenLayers - 1; i++) {
            network.add(new Layer(neuronsPerHidden, neuronsPerHidden, activator));
        }
        this.inputSize = inputSize;
        this.learningRate = learningRate;
        this.maxMomentum = maxMomentum;
        this.minMomentum = minMomentum;
    }

    //randomly reset all weights and biases
    private void reinitializeWeightsAndBiases(double scope) {
        for (int layer = 0; layer < network.size(); layer++) {
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                for (int weight = 0; weight < network.get(layer).get(neuron).getWeights().length(); weight++) {
                    network.get(layer).get(neuron).updateWeight(weight, Math.random() * scope * Math.pow(-1, (int)((Math.random())*10)));
                }
                network.get(layer).get(neuron).updateBias(Math.random() * scope * Math.pow(-1, (int)((Math.random())*10)));
            }
        }
    }

    //customized for sigmoid activations active in every neuron
    public ForwardPropOutput forwardProp(Vector input) {
        Vector[] matrix = new Vector[network.size()];
        Vector passed = network.get(0).activations(input);
        matrix[0] = passed.copy();
        for (int i = 1; i < network.size(); i++) {
            passed = network.get(i).activations(passed);
            matrix[i] = passed.copy();
        }
        return new ForwardPropOutput(passed, matrix);
    }

    //executes forward propagation starting from the specified layer
    public ForwardPropOutput forwardProp(Vector input, int startLayer) {
        Vector[] matrix = new Vector[network.size() - startLayer];
        Vector passed = network.get(startLayer).activations(input);
        matrix[0] = passed.copy();
        for (int i = 1; i < matrix.length; i++) {
            passed = network.get(i + startLayer).activations(passed);
            matrix[i] = passed.copy();
        }
        return new ForwardPropOutput(passed, matrix);
    }

    //implemented to validate the gradient calculated through backpropagation
    public double getApproximateLossDerivative(int layer, int neuron, int weight, Vector networkInput, Vector actual, double h) {

        ForwardPropOutput output = forwardProp(networkInput);
        Vector[] activations = output.getIntermediaryMatrix();

        ForwardPropOutput relativeOutput;
        if (layer == 0) {
            relativeOutput = forwardProp(networkInput, layer);
        } else {
            relativeOutput = forwardProp(activations[layer - 1], layer);
        }

        double initialLoss = computeLoss(relativeOutput.getResultant(), actual);


        Perceptron init = network.get(layer).get(neuron);
        init.getWeights().set(weight, init.getWeights().get(weight) + h);

        ForwardPropOutput changedOutput = null;
        if (layer == 0) {
            changedOutput = forwardProp(networkInput, layer);
        } else {
            changedOutput = forwardProp(activations[layer - 1], layer);
        }

        double finalLoss = computeLoss(changedOutput.getResultant(), actual);

        return (finalLoss - initialLoss) / h;
    }

    //training when recalculating loss is not feasible
    //TODO update to depend on gradient of respective dimension
    //TODO momentum to depend on concavity of respective dimension
    //TODO add weights visualizer
    public void train(ArrayList<NetworkData> trainingData, boolean visualize) {

        JFrame frame = null;
        JFreeChart chart = null;
        XYSeriesCollection dataset = null;
        XYSeries loss = null;
        XYSeries momentumTable = null;
        XYSeries gradientM = null;

        NetworkVisualizer visualizer = new NetworkVisualizer(this);
        visualizer.setVisible(true);

        if (visualize) {
            frame = new JFrame("Performance");
            dataset = new XYSeriesCollection();
            loss = new XYSeries("loss");
            momentumTable = new XYSeries("momentum");
            gradientM = new XYSeries("gradient");
            dataset.addSeries(loss);
            dataset.addSeries(momentumTable);
            dataset.addSeries(gradientM);
            chart = ChartFactory.createXYLineChart("Performance", "Epoch", "Y", dataset);
            frame.setContentPane(new ChartPanel(chart));
            frame.pack();
            frame.setVisible(true);
        }

        double momentum = maxMomentum;
        double epoch = 0;
        double lossf;
        while (momentum > minMomentum) {

            NetworkGradient cumulativeLossGradient = new NetworkGradient();
            for (NetworkData data : trainingData) {
                cumulativeLossGradient.add(getGradient(data.getInput(), data.getOutput()));
            }

            NetworkGradient updateGradient = getUpdateVector(cumulativeLossGradient);
            updateWeightsAndBiases(updateGradient);

            double gradientMagnitude = getGradientMagnitude(cumulativeLossGradient);
            momentum = 2 * maxMomentum * (Sigmoid.sigmoid(gradientMagnitude) - 0.5);
            lossf = cumulativeLoss(trainingData, this);

            epoch++;

            if (visualize) {
                loss.add(epoch, lossf);
                momentumTable.add(epoch, momentum);
                gradientM.add(epoch, gradientMagnitude);
            }

            //manage weights visualizer
            for (int layer = 0; layer < network.size(); layer++) {
                for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                    visualizer.set(neuron, layer, network.get(layer).get(neuron).getWeights().toString());
                }
            }
        }
    }

    //given a gradient, calculates the magnitude of that gradient
    public double getGradientMagnitude(NetworkGradient networkOutput) {
        double magnitude = 0;
        for (ArrayList<Vector> layer : networkOutput.getdLossdWeights()) {
            for (Vector neuron : layer) {
                for (double weight : neuron.getVector()) {
                    magnitude += weight * weight;
                }
            }
        }
        for (ArrayList<Double> layer : networkOutput.getdLossdBiases()) {
            for (Double dlossdbias : layer) {
                magnitude += dlossdbias * dlossdbias;
            }
        }
        return Math.sqrt(magnitude);
    }

    //given an update vector, updates this network accordingly
    public void updateWeightsAndBiases(NetworkGradient updateVector) {
        for (int layer = 0; layer < network.size(); layer++) {
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {

                Perceptron perceptron = network.get(layer).get(neuron);

                for (int weight = 0; weight < perceptron.getWeights().length(); weight++) {
                    perceptron.getWeights().set(weight, perceptron.getWeights().get(weight) + updateVector.getdLossdWeights().get(layer).get(neuron).get(weight));
                }
                perceptron.updateBias(perceptron.getBias() + updateVector.getdLossdBiases().get(layer).get(neuron));
            }
        }
    }

    //converts a loss vector to an update vector which can be directly applied to the network's weights and bias vector
    public NetworkGradient getUpdateVector(NetworkGradient gradient) {

        ArrayList<ArrayList<Vector>> dLossdWeights = gradient.getdLossdWeights();
        ArrayList<ArrayList<Double>> dLossdBiases = gradient.getdLossdBiases();

        ArrayList<ArrayList<Vector>> weightUpdates = new ArrayList<>();
        ArrayList<ArrayList<Double>> biasUpdates = new ArrayList<>();

        for (int layer = 0; layer < network.size(); layer++) {
            ArrayList<Vector> thisLayerW = new ArrayList<>();
            ArrayList<Double> thisLayerB = new ArrayList<>();
            for (int neuron = 0; neuron < dLossdWeights.get(layer).size(); neuron++) {

                Vector updates = new Vector(dLossdWeights.get(layer).get(neuron).length());
                for (int weight = 0; weight < updates.length(); weight++) {
//                    if (dLossdWeights.get(layer).get(neuron).get(weight) < 0) {
//                        updates.set(weight, learningRate * momentum);
//                    } else if (dLossdWeights.get(layer).get(neuron).get(weight) > 0) {
//                        updates.set(weight, -learningRate * momentum);
//                    } else {
//                        updates.set(weight, Math.pow(-1, (int) (Math.random() * 10)) * learningRate * momentum);
//                    }
                    updates.set(weight, -1 * learningRate * dLossdWeights.get(layer).get(neuron).get(weight));
                }
                thisLayerW.add(updates);

//                if (dLossdBiases.get(layer).get(neuron) < 0) {
//                    thisLayerB.add(learningRate * momentum);
//                } else if (dLossdBiases.get(layer).get(neuron) > 0) {
//                    thisLayerB.add(-learningRate * momentum);
//                } else {
//                    thisLayerB.add(Math.pow(-1, (int) (Math.random() * 10)) * learningRate * momentum);
//                }
                thisLayerB.add(-1 * learningRate * dLossdBiases.get(layer).get(neuron));
            }
            weightUpdates.add(thisLayerW);
            biasUpdates.add(thisLayerB);
        }
        return new NetworkGradient(weightUpdates, biasUpdates);
    }

    //for the specified input-output pair, calculates the derivative of loss with respect to each weight and bias
    public NetworkGradient getGradient(Vector input, Vector correct) {

        //all information needed to calculate necessary partial derivatives
        ForwardPropOutput output = forwardProp(input);

        Vector prediction = output.getResultant();
        Vector[] neuronActivations = output.getIntermediaryMatrix();

        //derivatives of the loss with respect to every activation of the last layer
        Vector dLossdLastLayer = getLossDWRTLastLayer(correct, neuronActivations[neuronActivations.length - 1]);

        //derivatives of each each neuron in layer n to each neuron in layer n-1
        ArrayList<ArrayList<Vector>> dLayersdPreviousLayers = getLayerDerivatives(neuronActivations);

        //derivatives of each activation with respect to the weights of that neuron
        ArrayList<ArrayList<Vector>> dActivationsdWeights = getWeightDerivatives(neuronActivations, input);

        //derivatives of each activation with respect to the weights of that neuron
        ArrayList<ArrayList<Double>> dActivationsdBias = getBiasDerivatives(neuronActivations, input);

        //derivatives of the loss value with respect to each activation
        Vector[] dLossdActivations = getLossDWRTactivations(dLossdLastLayer, dLayersdPreviousLayers);

        //derivatives of the loss with respect to each weight - not verified yet
        ArrayList<ArrayList<Vector>> dLossdWeights = getWeightDWRTLoss(dLossdActivations, dActivationsdWeights);

        //derivatives of the loss with respect to each bias - not verified yet
        ArrayList<ArrayList<Double>> dLossdBiases = getBiasDWRTLoss(dLossdActivations, dActivationsdBias);

        return new NetworkGradient(dLossdWeights, dLossdBiases);
    }

    //finds the derivative of each neuron's activation with respect to every weight within the neuron.
    public ArrayList<ArrayList<Vector>> getWeightDerivatives(Vector[] neuronActivations, Vector input) {
        ArrayList<ArrayList<Vector>> weightDerivatives = new ArrayList<>();
        for (int layer = network.size() - 1; layer >= 0; layer--) {
            ArrayList<Vector> thisLayer = new ArrayList<>();
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {

                Vector pDerivatives;
                double guess;
                if (layer == 0) {

                    pDerivatives = new Vector(input.length());
                    guess = network.get(layer).get(neuron).unactivatedGuess(input);

                    for (int weight = 0; weight < input.length(); weight++) {
                        double factor = input.get(weight);
                        pDerivatives.set(weight, network.get(layer).get(neuron).getActivator().activationDerivative(guess) * factor);
                    }

                } else {

                    pDerivatives = new Vector(network.get(layer - 1).length());
                    guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);

                    for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                        double factor = neuronActivations[layer - 1].get(weight);
                        pDerivatives.set(weight, network.get(layer).get(neuron).getActivator().activationDerivative(guess) * factor);
                    }
                }
                thisLayer.add(pDerivatives);
            }
            weightDerivatives.add(0, thisLayer);
        }
        return weightDerivatives;
    }

    //Returns a vector matrix, where each matrix position corresponds to a neuron location
    //Vector indices in layer l line in index with the neurons of layer n - 1, in terms of of their derivatives
    //Specifically, at [2][0][3] is stored the derivative of the activation of neuron [2][0] with respect to the activation [1][3]
    //The first stored row is marked null to indicate that derivatives with respect to input values are obsolete
    public ArrayList<ArrayList<Vector>> getLayerDerivatives(Vector[] neuronActivations) {
        ArrayList<ArrayList<Vector>> layerDerivatives = new ArrayList<>();
        for (int layer = network.size() - 1; layer >= 1; layer--) {
            ArrayList<Vector> thisLayer = new ArrayList<>();
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                Vector pDerivatives = new Vector(network.get(layer - 1).length());
                double guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);
                for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                    double factor = network.get(layer).get(neuron).getWeights().get(weight);
                    pDerivatives.set(weight, network.get(layer).get(neuron).getActivator().activationDerivative(guess) * factor);
                }
                thisLayer.add(pDerivatives);
            }
            layerDerivatives.add(0, thisLayer);
        }
        ArrayList<Vector> initialLayer = new ArrayList<>();
        for (int i = 0; i < network.get(0).length(); i++) {
            initialLayer.add(null);
        }
        layerDerivatives.add(0, initialLayer);
        return layerDerivatives;
    }

    //returns the derivative of each activation with respect to the bias associated with that same activation (matched by index)
    public ArrayList<ArrayList<Double>> getBiasDerivatives(Vector[] neuronActivations, Vector input) {
        ArrayList<ArrayList<Double>> biasDerivatives = new ArrayList<>();
        for (int layer = network.size() - 1; layer >= 0; layer--) {
            ArrayList<Double> thisLayer = new ArrayList<>();
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                double bderivative;
                if (layer == 0) {
                    bderivative = network.get(layer).get(neuron).getActivator().activationDerivative(network.get(layer).get(neuron).unactivatedGuess(input));
                } else {
                    bderivative = network.get(layer).get(neuron).getActivator().activationDerivative(network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]));
                }
                thisLayer.add(bderivative);
            }
            biasDerivatives.add(0, thisLayer);
        }
        return biasDerivatives;
    }

    //derivative of loss with respect to each each activation in the last layer
    public Vector getLossDWRTLastLayer(Vector actual, Vector prediction) { //derivatives calculated for MSE
        Vector out = new Vector(actual.length());
        for (int i = 0; i < out.length(); i++) {
            out.set(i, -1 * (actual.get(i) - prediction.get(i)));
        }
        return out;
    }

    //returns a matrix representing the derivative of loss with respect to each activation at the current network configuration
    public Vector[] getLossDWRTactivations(Vector lastLayerDerivatives, ArrayList<ArrayList<Vector>> layerDerivatives) {
        Vector[] activationDerivatives = new Vector[network.size()];
        activationDerivatives[network.size() - 1] = lastLayerDerivatives.copy();
        for (int layer = network.size() - 1; layer >= 1; layer--) {
            Vector pDerivatives = new Vector(network.get(layer - 1).length());
            for (int neuron2 = 0; neuron2 < network.get(layer - 1).length(); neuron2++) {
                double completeDerivative = 0;
                for (int neuron1 = 0; neuron1 < network.get(layer).length(); neuron1++) {
                    completeDerivative += layerDerivatives.get(layer).get(neuron1).get(neuron2) * lastLayerDerivatives.get(neuron1);
                }
                pDerivatives.set(neuron2, completeDerivative);
            }
            lastLayerDerivatives = pDerivatives.copy();
            activationDerivatives[layer - 1] = pDerivatives;
        }
        return activationDerivatives;
    }

    //finds the derivatives of loss with respect to each weight
    public ArrayList<ArrayList<Vector>> getWeightDWRTLoss(Vector[] activationDWRTLoss, ArrayList<ArrayList<Vector>> weightDWRTactivations) {
        ArrayList<ArrayList<Vector>> weightDWRTLoss = new ArrayList<>();
        for (int layer = 0; layer < network.size(); layer++) {
            ArrayList<Vector> thisLayer = new ArrayList<>();
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                Vector weightDerivatives;
                if (layer == 0) {
                    weightDerivatives = new Vector(inputSize);
                    for (int weight = 0; weight < inputSize; weight++) {
                        weightDerivatives.set(weight, activationDWRTLoss[layer].get(neuron) * weightDWRTactivations.get(layer).get(neuron).get(weight));
                    }
                } else {
                    weightDerivatives = new Vector(network.get(layer).get(neuron).getWeights().length());
                    for (int weight = 0; weight < weightDerivatives.length(); weight++) {
                        weightDerivatives.set(weight, activationDWRTLoss[layer].get(neuron) * weightDWRTactivations.get(layer).get(neuron).get(weight));
                    }
                }
                thisLayer.add(weightDerivatives);
            }
            weightDWRTLoss.add(thisLayer);
        }
        return weightDWRTLoss;
    }

    //finds the derivative of loss with respect to each bias
    public ArrayList<ArrayList<Double>> getBiasDWRTLoss(Vector[] activationDWRTLoss, ArrayList<ArrayList<Double>> biasDWRTactivations) {
        ArrayList<ArrayList<Double>> biasDWRTLoss = new ArrayList<>();
        for (int layer = 0; layer < network.size(); layer++) {
            ArrayList<Double> thisLayer = new ArrayList<>();
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                thisLayer.add(biasDWRTactivations.get(layer).get(neuron) * activationDWRTLoss[layer].get(neuron));
            }
            biasDWRTLoss.add(thisLayer);
        }
        return biasDWRTLoss;
    }

    //computes the total loss given the training set being used
    public static double cumulativeLoss(ArrayList<NetworkData> testSet, NeuralNetwork neuralNetwork) {
        double out = 0;
        for (NetworkData data : testSet) {
            out += computeLoss(neuralNetwork.forwardProp(data.getInput()).getResultant(), data.getOutput());
        }
        return out;
    }

    //computes the loss given a specific input-output pairing
    public static double computeLoss(Vector predicted, Vector actual) {
        double sum = 0;
        assert predicted.length() == actual.length();
        for (int i = 0; i < predicted.length(); i++) {
            sum += (predicted.get(i) - actual.get(i)) * (predicted.get(i) - actual.get(i));
        }
        return sum * 0.5;
    }

    public Layer getLayer(int l) {return network.get(l);}

    public int largestLayerSize() {
        int max = 0;
        for (Layer layer : network) {
            if (layer.length() > max) {
                max = layer.length();
            }
        }
        return max;
    }

    public int numLayers() {
        return network.size();
    }

    //visualizer for the network
    public void displayWeightsAndBiases() {
        for (int i = 0; i < 5; i++) System.out.println();
        System.out.println("DISPLAYING WEIGHTS");
        for (int layer = 0; layer < network.size(); layer++) {
            System.out.println("---------------------------");
            System.out.println("LAYER: " + layer);
            System.out.println("---------------------------");
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                System.out.println("NEURON: " + neuron);
                System.out.println("WEIGHTS: " + network.get(layer).get(neuron).getWeights());
                System.out.println("BIAS: " + network.get(layer).get(neuron).getBias());
            }
        }
    }

    //NetworkGradients represent loss vectors which describe both the weights and biases of a network
    public static class NetworkGradient {

        private ArrayList<ArrayList<Vector>> dLossdWeights;
        private ArrayList<ArrayList<Double>> dLossdBiases;

        public NetworkGradient(ArrayList<ArrayList<Vector>> dLossdWeights, ArrayList<ArrayList<Double>> dLossdBiases) {
            this.dLossdBiases = dLossdBiases;
            this.dLossdWeights = dLossdWeights;
        }

        public NetworkGradient() {
        }

        public void add(NetworkGradient other) {
            if (dLossdWeights == null || dLossdBiases == null) {
                dLossdWeights = other.getdLossdWeights();
                dLossdBiases = other.getdLossdBiases();
            } else {
                for (int layer = 0; layer < dLossdWeights.size(); layer++) {
                    for (int neuron = 0; neuron < dLossdWeights.get(layer).size(); neuron++) {
                        this.dLossdBiases.get(layer).set(neuron, this.dLossdBiases.get(layer).get(neuron) + other.dLossdBiases.get(layer).get(neuron));
                        this.dLossdWeights.get(layer).get(neuron).add(other.dLossdWeights.get(layer).get(neuron));
                    }
                }
            }
        }

        public ArrayList<ArrayList<Vector>> getdLossdWeights() {
            return dLossdWeights;
        }

        public ArrayList<ArrayList<Double>> getdLossdBiases() {
            return dLossdBiases;
        }
    }

    public static class ForwardPropOutput {

        private final Vector resultant;
        private final Vector[] intermediaryMatrix;

        public ForwardPropOutput(Vector resultant, Vector[] intermediaryMatrix) {
            this.resultant = resultant;
            this.intermediaryMatrix = intermediaryMatrix;
        }

        public Vector getResultant() {
            return resultant;
        }

        public Vector[] getIntermediaryMatrix() {
            return intermediaryMatrix;
        }
    }

}
