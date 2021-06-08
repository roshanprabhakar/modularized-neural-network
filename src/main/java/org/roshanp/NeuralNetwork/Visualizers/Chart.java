package org.roshanp.NeuralNetwork.Visualizers;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataItem;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.roshanp.NeuralNetwork.NeuralNetwork;

import javax.swing.*;
import java.util.ArrayList;
import java.util.HashMap;

public class Chart extends JFrame {

    private NeuralNetwork network;
    private XYSeriesCollection collections;

    private XYDataItem latest = null;

    public Chart(String name, String x, String y, NeuralNetwork network) {

        collections = new XYSeriesCollection();
        this.network = network;

        JFreeChart chart = ChartFactory.createScatterPlot(name, x, y, collections);
        setContentPane(new ChartPanel(chart));

        collections.addSeries(new XYSeries("last entry"));

        pack();
    }

    public void addSeries(String name) {
        collections.addSeries(new XYSeries(name));
    }

    public void update(String series, double x, double y) {
        XYDataItem item = new XYDataItem(x, y);
        if (latest == null) {
            collections.getSeries("last entry").add(item);
        } else {
            collections.getSeries("last entry").remove(0);
            collections.getSeries("last entry").add(item);
            collections.getSeries(series).add(latest);
        }
        latest = item;
    }
}
