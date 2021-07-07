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
import java.util.Arrays;
import java.util.HashMap;

public class Chart extends JFrame {

    private XYSeriesCollection collections;
    private String series;

    private XYDataItem latest = null;

    public Chart(String name, String x, String y, String series) {

        collections = new XYSeriesCollection();
        this.series = series;

        JFreeChart chart = ChartFactory.createScatterPlot(name, x, y, collections);
        setContentPane(new ChartPanel(chart));

        collections.addSeries(new XYSeries("last entry"));
        collections.addSeries(new XYSeries(series));

        pack();

    }

    public void update(double x, double y) {

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
