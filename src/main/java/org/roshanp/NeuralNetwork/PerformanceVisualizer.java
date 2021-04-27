package org.roshanp.NeuralNetwork;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;

public class PerformanceVisualizer extends JFrame {

    private final XYSeries loss;
    private final XYSeries gradientM;
    private final XYSeries accuracy;

    private int epoch = 0;

    public PerformanceVisualizer(NeuralNetwork network) {
        super("Performance Visualizer");

        XYSeriesCollection dataset = new XYSeriesCollection();

        loss = new XYSeries("loss");
        gradientM = new XYSeries("gradient");
        accuracy = new XYSeries("accuracy");

        dataset.addSeries(loss);
        dataset.addSeries(gradientM);
        dataset.addSeries(accuracy);

        JFreeChart chart = ChartFactory.createXYLineChart("Performance", "Epoch", "Y", dataset);

        this.setContentPane(new ChartPanel(chart));
        this.pack();
    }

    public void update(double loss, double gradient, double accuracy) {
        epoch++;
        this.loss.add(epoch, loss);
        this.gradientM.add(epoch, gradient);
        this.accuracy.add(epoch, accuracy);
    }
}
