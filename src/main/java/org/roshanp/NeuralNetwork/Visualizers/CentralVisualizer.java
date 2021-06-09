package org.roshanp.NeuralNetwork.Visualizers;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

public class CentralVisualizer extends JFrame {

    private JFrame[][] REF;

    public CentralVisualizer(ArrayList<JFrame> frames, int rows, int cols) {
        super("visualizer");

        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(rows, cols));

        REF = new JFrame[rows][cols];

        for (int i = 0; i < frames.size(); i++) {

            int col = i % cols;
            int row = (i - col) / cols;

            REF[row][col] = frames.get(i);
            panel.add(frames.get(i).getContentPane());
        }

        this.setContentPane(panel);
        this.pack();

    }


}
