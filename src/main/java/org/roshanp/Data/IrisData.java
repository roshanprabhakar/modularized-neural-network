package org.roshanp.Data;

import org.roshanp.NeuralNetwork.NetworkData;
import org.roshanp.NeuralNetwork.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class IrisData extends NetworkData {

    private static final HashMap<String, Vector> map = new HashMap<>() {
        {
            put("Iris-setosa", new Vector(new double[]{1, 0, 0}));
            put("Iris-versicolor", new Vector(new double[]{0, 1, 0}));
            put("Iris-virginica", new Vector(new double[]{0, 0, 1}));
        }
    };

    private IrisData(String data) {
        super(getIn(data), getOut(data));
    }

    public static ArrayList<NetworkData> loadIrisData(String filepath) {
        ArrayList<NetworkData> out = new ArrayList<>();
        BufferedReader br;
        try {
            br = new BufferedReader(new FileReader(new File(filepath)));
            String nline;
            while ((nline = br.readLine()) != null) {
                out.add(new IrisData(nline));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return out;
    }

    private static Vector getIn(String l) {
        Vector out = new Vector(4);
        String[] p = l.split(",");
        for (int i = 0; i < out.length(); i++) {
            out.set(i, Double.parseDouble(p[i]));
        }
        return out;
    }

    private static Vector getOut(String l) {
        return map.get(l.split(",")[4]);
    }

}
