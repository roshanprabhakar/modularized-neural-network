package org.roshanp.NeuralNetwork.Visualizers;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.roshanp.NeuralNetwork.NeuralNetwork;

import javax.swing.*;
import java.util.ArrayList;
import java.util.HashMap;

public class Chart extends JFrame {

    private NeuralNetwork network;
    private XYSeriesCollection collections;

    public Chart(String name, String x, String y, NeuralNetwork network) {

        collections = new XYSeriesCollection();
        this.network = network;

        JFreeChart chart = ChartFactory.createScatterPlot(name, x, y, collections);
        setContentPane(new ChartPanel(chart));



        pack();
    }

    public void addSeries(String name) {
        collections.addSeries(new XYSeries(name));
    }

    public void update(String series, double x, double y) {
        collections.getSeries(series).add(x, y);
    }
}
