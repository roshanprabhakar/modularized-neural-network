package org.roshanp.NeuralNetwork;

import javax.swing.*;
import java.awt.*;

public class NetworkVisualizer extends JFrame {

    private int rows;
    private int cols;

    private JTextField[][] REF;

    private NeuralNetwork network;

    public NetworkVisualizer(NeuralNetwork network) {
        super("network visualizer");

        rows = network.largestLayerSize();
        cols = network.numLayers();

        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(rows, cols));

        REF = new JTextField[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                JTextField field = new JTextField("--");
                REF[r][c] = field;
                panel.add(field);
            }
        }

        this.network = network;

        this.getContentPane().add(panel, BorderLayout.CENTER);
        this.pack();
    }

    public void set(int r, int c, String text) {
        REF[r][c].setText(text);
    }

    public void update() {
        for (int layer = 0; layer < network.numLayers(); layer++) {
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                set(neuron, layer, network.get(layer).get(neuron).getWeights().toString());
            }
        }
    }
}
