package org.roshanp.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

public class DataVisualizer extends JFrame {

    //loads 20 data points
    private static final int rows = 10;
    private static final int cols = 4;

    private ArrayList<NetworkData> data;
    private NeuralNetwork network;

    JTextField[][] REF;

    public DataVisualizer(ArrayList<NetworkData> data, NeuralNetwork network) {
        super("data visualizer");
        this.data = data;
        this.network = network;

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

        this.getContentPane().add(panel, BorderLayout.CENTER);
        this.pack();

        reload();
    }

    public void reload() {
        for (int i = 1; i < rows; i++) {
            NetworkData d1 = data.get(i);
            NeuralNetwork.ForwardPropOutput out = network.forwardProp(d.getInput());
            REF[i][1]
        }
    }
}
