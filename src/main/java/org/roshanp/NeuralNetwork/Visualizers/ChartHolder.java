package org.roshanp.NeuralNetwork.Visualizers;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

public class ChartHolder extends JFrame {

    private Chart[][] REF;

    public ChartHolder(ArrayList<Chart> charts, int rows, int cols) {
        super("visualizer");

        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(rows, cols));

        REF = new Chart[rows][cols];

        for (int i = 0; i < charts.size(); i++) {

            int col = i % cols;
            int row = (i - col) / cols;

            REF[row][col] = charts.get(i);
            panel.add(charts.get(i).getContentPane());
        }

        this.setContentPane(panel);
        this.pack();

    }


}
