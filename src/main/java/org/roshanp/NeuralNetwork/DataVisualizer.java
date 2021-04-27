package org.roshanp.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.Collections;

public class DataVisualizer extends JFrame {

    //loads 20 data points
    private static final int rows = 10;
    private static final int cols = 3;

    private ArrayList<NetworkData> data;
    private NeuralNetwork network;

    private JTextField[][] REF;

    public DataVisualizer(ArrayList<NetworkData> data, NeuralNetwork network) {
        super("data visualizer");

        Collections.shuffle(data);

        this.data = data;
        this.network = network;

        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(rows, cols));

        REF = new JTextField[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                JTextField field = null;
                if (r == 0) {
                    if (c == 0) {
                        field = new JTextField("input");
                    } else if (c == 1) {
                        field = new JTextField("actual");
                    } else {
                        field = new JTextField("guess");
                    }
                } else {
                    field = new JTextField("--");
                }
                REF[r][c] = field;
                panel.add(field);
            }
        }

        for (int r = 1; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (c == 0) {
                    REF[r][c].setText(data.get(r).getInput().toString());
                } else if (c == 1) {
                    REF[r][c].setText(data.get(r).getOutput().toString());
                }
            }
        }

        this.getContentPane().add(panel, BorderLayout.CENTER);
        this.pack();

        reload();
    }

    public void reload() {
        for (int i = 1; i < rows; i++) {
            NetworkData d = data.get(i);
            NeuralNetwork.ForwardPropOutput out = network.forwardProp(d.getInput());
            REF[i][2].setText(out.getResultant().toString());
        }
    }
}
