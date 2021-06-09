package org.roshanp.NeuralNetwork;

import org.roshanp.NeuralNetwork.Activations.Activator;
import org.roshanp.NeuralNetwork.Activations.Sigmoid;
import org.roshanp.NeuralNetwork.Visualizers.Chart;
import org.roshanp.NeuralNetwork.Visualizers.ChartHolder;

import java.util.ArrayList;
import java.util.Arrays;

//IMPORTANT
//all data must be bounded in input and output by the scope of the activation function (layers 0 and n are activated by the specified activation, therefore bounded)
public class NeuralNetwork {

    public static final Activator sigmoid = new Sigmoid();

    //primary network object
    private ArrayList<Layer> network;

    //input data dimension
    private int inputSize;

    //learning parameters
    private double g; //gravitational acceleration
    private double initial_velocity; //initial step size

    /**
     * @param layers    layers[i] corresponds to the number of neurons at layer i in the network, where i is the layer index AFTER input
     * @param inputSize input dimension
     * @param g         magnitude of gravitational acceleration driving the weight updates
     * @param vi        initial velocity of training decent
     * @param activator type of activation function applied at every neuron
     */
    public NeuralNetwork(int[] layers, int inputSize, double g, double vi, Activator activator, boolean randominit) {
        network = new ArrayList<>();
        network.add(new Layer(layers[0], inputSize, activator, randominit));
        for (int layer = 1; layer < layers.length; layer++) {
            network.add(new Layer(layers[layer], layers[layer - 1], activator, randominit));
        }
        this.inputSize = inputSize;
        this.g = g;
        this.initial_velocity = vi;
    }

    /**
     * @param inputSize        input dimension
     * @param neuronsPerHidden constant neuron count among all layers
     * @param numHiddenLayers  number of layers in the network, including the output layer
     * @param g                gravitational acceleration driving weight updates
     * @param vi               initial velocity of training decent
     * @param activator        the activator applied at every neuron
     */
    public NeuralNetwork(int inputSize, int neuronsPerHidden, int numHiddenLayers, double g, double vi, Activator activator, boolean randominit) {
        network = new ArrayList<>();
        network.add(new Layer(neuronsPerHidden, inputSize, activator, randominit));
        for (int i = 0; i < numHiddenLayers - 1; i++) {
            network.add(new Layer(neuronsPerHidden, neuronsPerHidden, activator, randominit));
        }
        this.inputSize = inputSize;
        this.g = g;
        this.initial_velocity = vi;
    }


    //TODO pause/resume training functionality (multi-threaded run + observing controllers)
    //TODO implement training medium viscosity
    public void train(ArrayList<NetworkData> trainingData, boolean verbose) throws InterruptedException {

        Chart lossChart = null;
        Chart velocityChart = null;
        Chart accelChart = null;

        if (verbose) {
            lossChart = new Chart("Position", "weight", "loss", this);
            lossChart.addSeries("weight");

            velocityChart = new Chart("Velocity", "epoch", "Velocity", this);
            velocityChart.addSeries("velocity");

            accelChart = new Chart("Acceleration", "epoch", "Acceleration", this);
            accelChart.addSeries("acceleration");

            ChartHolder holder = new ChartHolder(new ArrayList<>(Arrays.asList(lossChart, velocityChart, accelChart)), 2, 2);
            holder.setVisible(true);
        }

        double epoch = 0;
        double lossf;
        double accuracy;

        NetworkGradient velocity = new NetworkGradient(this, initial_velocity);

        //initial iteration: step size is initial velocity
        NetworkGradient previousLossGradient = new NetworkGradient();
        for (NetworkData data : trainingData) {
            previousLossGradient.add(getGradient(data.getInput(), data.getOutput()));
        }

        double lossi;

        //initialize weight position
        network.get(0).get(0).getWeights().set(0, 0.1);
        velocity.dLossdWeights.get(0).get(0).set(0, initial_velocity);

        while (true) {

            NetworkGradient cumulativeLossGradient = new NetworkGradient();
            for (NetworkData data : trainingData) {
                cumulativeLossGradient.add(getGradient(data.getInput(), data.getOutput()));
            }

            updateWeightsAndBiases(velocity, 0, 0, 0);

            double gradientMagnitude = getGradientMagnitude(cumulativeLossGradient);
            lossf = cumulativeLoss(trainingData, this);
//            accuracy = getAccuracy(trainingData, this);

            if (verbose) {
               lossChart.update("weight", network.get(0).get(0).get(0), lossf);
               velocityChart.update("velocity", epoch, velocity.dLossdWeights.get(0).get(0).get(0));
               accelChart.update("acceleration", epoch, a(cumulativeLossGradient.dLossdWeights.get(0).get(0).get(0)));
            }

            updateVelocity(velocity, previousLossGradient, cumulativeLossGradient);

            previousLossGradient = cumulativeLossGradient;

            epoch++;
            Thread.sleep(800);

        }
    }


    /*
     * Update rule to mimic calculus behind real world physical accumulated effect of acceleration
     * where A = network parallel of acceleration due to gravity,
     * where a = acceleration along the dimension of weight n
     * where v = accumulated velocity
     * where w = position in weight space
     * where L(x) = loss as a function of weights x
     *
     * a = A/2 * sin(-2 * atan(1/L'(w)))
     * [integral wi -> wf](a(w))[dw] + vi^2/2 = vf^2/2
     * */

    //converts a loss vector to an update vector which can be directly applied to the network's weights and bias vector
    //update function: dnetwork = velocity across a single epoch

    /**
     * @param velocity velocity across every weight and bias dimension
     */
    public void updateVelocity(NetworkGradient velocity, NetworkGradient mi, NetworkGradient mf) {

        double vi = velocity.dLossdWeights.get(0).get(0).get(0);
        double wi = network.get(0).get(0).get(0);
        double wf = network.get(0).get(0).get(0) + vi;
        double wpi = mi.dLossdWeights.get(0).get(0).get(0);
        double wpf = mf.dLossdWeights.get(0).get(0).get(0);
        double wom = 0.5 * (a(wpi) + a(wpf)) * vi; //trapezoidal integral
        double kf = wom + 0.5 * (vi * vi);
        double df = (vi / Math.abs(vi)) * (kf / Math.abs(kf));
        double vf = Math.sqrt(2 * Math.abs(kf)) * df;

        velocity.dLossdWeights.get(0).get(0).set(0, vf);

//        for (int layer = 0; layer < network.size(); layer++) {
//            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
//                for (int weight = 0; weight < network.get(layer).get(neuron).getWeights().length(); weight++) {
//                    velocity.dLossdWeights.get(layer).get(neuron).set(weight, v(
//                            velocity.dLossdWeights.get(layer).get(neuron).get(weight),
//                            olddLdWB.dLossdWeights.get(layer).get(neuron).get(weight),
//                            dLdWB.dLossdWeights.get(layer).get(neuron).get(weight),
//                            velocity.dLossdWeights.get(layer).get(neuron).get(weight)
//                    ));
//                }
//                velocity.dLossdBiases.get(layer).set(neuron, v(
//                        velocity.dLossdBiases.get(layer).get(neuron),
//                        olddLdWB.dLossdBiases.get(layer).get(neuron),
//                        dLdWB.dLossdBiases.get(layer).get(neuron),
//                        velocity.dLossdBiases.get(layer).get(neuron)
//                ));
//            }
//        }

    }

    /**
     * @param wp slope of loss against weight in the considered dimension
     * @return acceleration in current weight space
     */
    private double a(double wp) {
        return g / 2.0 * Math.sin(-2 * Math.atan(1 / wp));
    }

    /**
     * @param vh  velocity at previous iteration
     * @param wph weight derivative at previous iteration
     * @param wpi weight derivative at current iteration
     * @param dw  change in weight from previous to current iterations
     * @return current velocity
     * <p>
     * in the implemented update, dw = vh
     */
    private double v(double vh, double wph, double wpi, double dw) {

        double ai = a(wpi);
        double ah = a(wph);

        double a = (ai * (ai / ah) + 1) / 2;
        double b = (dw * ai) / (2 * ((ai / ah) + 1));

        double dv;
        if (ah < 0 && ai > 0) {
            dv = a - b;
        } else if (ah > 0 && ai < 0) {
            dv = b - a;
        } else {
            dv = ((dw) / 2.0) * (ah + ai);
        }


        double vi = vh * vh + dv;
        if (vi > 0) {
            return Math.sqrt(vi);
        } else if (vi < 0) {
            return -1 * Math.sqrt(vi * -1);
        } else {
            return initial_velocity;
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

    //
    //given an update vector, updates this network accordingly
    public void updateWeightsAndBiases(NetworkGradient updateVector, int layer, int neuron, int weight) {
//        for (int layer = 0; layer < network.size(); layer++) {
//            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
//
//                Perceptron perceptron = network.get(layer).get(neuron);
//
//                for (int weight = 0; weight < perceptron.getWeights().length(); weight++) {
//                    perceptron.getWeights().set(weight, perceptron.getWeights().get(weight) + updateVector.getdLossdWeights().get(layer).get(neuron).get(weight));
//                }
//                perceptron.updateBias(perceptron.getBias() + updateVector.getdLossdBiases().get(layer).get(neuron));
//            }
//        }

        Perceptron p = network.get(layer).get(neuron);
        p.updateWeight(weight, p.get(weight) + updateVector.getdLossdWeights().get(layer).get(neuron).get(weight));
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
        ArrayList<ArrayList<Vector>> dLossdWeights = getLossDWRTWeights(dLossdActivations, dActivationsdWeights);

        //derivatives of the loss with respect to each bias - not verified yet
        ArrayList<ArrayList<Double>> dLossdBiases = getLossDWRTBiases(dLossdActivations, dActivationsdBias);

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
    public ArrayList<ArrayList<Vector>> getLossDWRTWeights(Vector[] activationDWRTLoss, ArrayList<ArrayList<Vector>> weightDWRTactivations) {
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
    public ArrayList<ArrayList<Double>> getLossDWRTBiases(Vector[] activationDWRTLoss, ArrayList<ArrayList<Double>> biasDWRTactivations) {
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


    //randomly reset all weights and biases
    private void reinitializeWeightsAndBiases(double scope) {
        for (int layer = 0; layer < network.size(); layer++) {
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                for (int weight = 0; weight < network.get(layer).get(neuron).getWeights().length(); weight++) {
                    network.get(layer).get(neuron).updateWeight(weight, Math.random() * scope * Math.pow(-1, (int) ((Math.random()) * 10)));
                }
                network.get(layer).get(neuron).updateBias(Math.random() * scope * Math.pow(-1, (int) ((Math.random()) * 10)));
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

    //computes the loss given a specific input-output pairing
    public static double computeLoss(Vector predicted, Vector actual) {
        double sum = 0;
        assert predicted.length() == actual.length();
        for (int i = 0; i < predicted.length(); i++) {
            sum += (predicted.get(i) - actual.get(i)) * (predicted.get(i) - actual.get(i));
        }
        return sum * 0.5;
    }

    public Layer get(int l) {
        return network.get(l);
    }

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

        public NetworkGradient(NeuralNetwork template) {

            ArrayList<ArrayList<Vector>> dLossdWeights = new ArrayList<>();
            ArrayList<ArrayList<Double>> dLossdBiases = new ArrayList<>();

            for (int l = 0; l < template.numLayers(); l++) {
                dLossdWeights.add(new ArrayList<>());
                for (int n = 0; n < template.get(l).length(); n++) {
                    dLossdWeights.get(l).add(new Vector(template.get(l).get(n).getWeights().length()));
                }
                dLossdBiases.add(new ArrayList<>());
                for (int i = 0; i < template.get(l).length(); i++) dLossdBiases.get(dLossdBiases.size() - 1).add(0.0);
            }

            this.dLossdWeights = dLossdWeights;
            this.dLossdBiases = dLossdBiases;
        }

        public NetworkGradient(NeuralNetwork template, double c) {
            this(template);
            for (int l = 0; l < template.numLayers(); l++) {
                for (int n = 0; n < template.get(l).length(); n++) {
                    for (int w = 0; w < template.get(l).get(n).getWeights().length(); w++) {
                        dLossdWeights.get(l).get(n).set(w, c);
                    }
                    dLossdBiases.get(l).set(n, c);
                }
            }
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

        public String toString() {
            StringBuilder out = new StringBuilder();
            for (int l = 0; l < dLossdWeights.size(); l++) {
                for (int n = 0; n < dLossdWeights.get(l).size(); n++) {
                    out.append(dLossdWeights.get(l).get(n)).append(" ").append(dLossdBiases.get(l).get(n)).append("    ");
                }
                out.append("\n");
            }
            return out.toString();
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

    public static double getAccuracy(ArrayList<NetworkData> trainingSet, NeuralNetwork network) {
        double correct = 0;
        for (NetworkData data : trainingSet) {
            Vector resultant = network.forwardProp(data.getInput()).getResultant();
            if (data.stepwise(resultant).equals(data.getOutput())) correct++;
        }
        return correct / trainingSet.size() * 100;
    }
}
