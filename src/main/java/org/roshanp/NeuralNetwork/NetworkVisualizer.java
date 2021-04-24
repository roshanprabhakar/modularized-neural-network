package org.roshanp.NeuralNetwork;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;

public class NetworkVisualizer extends JFrame {

    private int rows;
    private int cols;

    private JTextField[][] REF;

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

        this.getContentPane().add(panel, BorderLayout.CENTER);
        this.pack();
    }

    public void set(int r, int c, String text) {
        REF[r][c].setText(text);
    }


}