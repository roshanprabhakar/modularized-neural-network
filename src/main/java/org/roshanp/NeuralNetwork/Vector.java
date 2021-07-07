package org.roshanp.NeuralNetwork;

public class Vector {

    private double[] vector;

    public Vector(double[] vector) {
        this.vector = vector;
    }

    public Vector(int i) {
        this.vector = new double[i];
    }

    public Vector(int i, boolean random) {
        this.vector = new double[i];
        for (int j = 0; j < vector.length; j++) {
            vector[j] = Math.random() * 1;
        }
    }


    public int length() {
        return vector.length;
    }

    public double magnitude() {
        double s = 0;
        for (double i : vector) {
            s += i * i;
        }
        return Math.sqrt(s);
    }

    public double get(int i) {
        return vector[i];
    }

    public void set(int index, double value) {
        vector[index] = value;
    }


    public Vector multiplyScalar(double scalar) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= scalar;
        }
        return this;
    }

    public static Vector multiplyScalar(double scalar, Vector v) {
        Vector out = v.copy();
        for (int i = 0; i < v.length(); i++) {
            out.set(i, out.get(i) * scalar);
        }
        return out;
    }

    public Vector add(Vector other) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] += other.get(i);
        }
        return this;
    }

    public static Vector add(Vector v, Vector p) {
        Vector out = new Vector(v.length());
        for (int i = 0; i < v.length(); i++) {
            out.set(i, v.get(i) + p.get(i));
        }
        return out;
    }

    public Vector subtract(Vector other) {
        this.add(other.multiplyScalar(-1));
        return this;
    }

    public double dotProduct(Vector other) {
        return dotProduct(this, other);
    }

    public static double dotProduct(Vector v, Vector p) {
        double sum = 0;
        String out = "";
        for (int i = 0; i < v.length(); i++) {
            out += v.get(i) + " * " + p.get(i);
            if (i != v.length() - 1) out += " + ";
            sum += v.get(i) * p.get(i);
        }
//        System.out.println(out + " = " + sum);
        return sum;
    }

    public static double multiply(Vector v, Vector p) {
        return dotProduct(v, p);
    }

    public Vector expand(int power) {
        Vector out = new Vector(this.length() * power);
        for (int i = 0; i < length(); i++) {
            for (int pow = 1; pow <= power; pow++) {
                out.set(i * power + pow - 1, Math.pow(this.get(i), pow));
            }
        }

        return out;
    }

    public Vector raise(double p) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = Math.pow(vector[i], p);
        }
        return this;
    }

    public Vector square() {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = vector[i] * vector[i];
        }
        return this;
    }

    public double loss(Vector correct) {
        if (correct.length() != this.length()) return Double.MAX_VALUE;
        double loss = 0;
        for (int i = 0; i < this.length(); i++) {
            loss += (correct.get(i) - this.get(i)) * (correct.get(i) - this.get(i));
        }
        return loss * (0.5);
    }

    public Vector copy() {
        return new Vector(vector.clone());
    }

    public double[] getVector() {
        return vector;
    }

    public String toString() {
        StringBuilder out = new StringBuilder("(");
        for (int i = 0; i < vector.length; i++) {
            out.append(NetworkData.round(vector[i], 1000));
//            out.append(vector[i]);
            if (i != vector.length - 1) out.append(", ");
        }
        out.append(")");
        return out.toString();
    }

    public Vector getNetworkOutputVector() {
        double max = Integer.MIN_VALUE;
        for (double value : this.getVector()) {
            if (value > max) max = value;
        }
        Vector out = new Vector(this.length());
        for (int i = 0; i < this.length(); i++) {
            if (this.get(i) == max) {
                out.set(i, 1);
            }
        }
        return out;
    }

    public boolean equals(Vector other) {
        assert length() == other.length();
        for (int i = 0; i < other.length(); i++) {
            if (get(i) != other.get(i)) return false;
        }
        return true;
    }

    //********************
    //2D VECTOR OPERTATION
    //********************

    public Vector(double magnitude, double theta) {
        this(new double[]{magnitude, 0});
        rotateTo(theta);
    }

    public void rotateTo(double theta) {
        theta = Math.PI - theta;
        assert length() == 2;
        double m = magnitude();
        set(1, m * Math.cos(theta));
        set(0, m * Math.sin(theta));
    }

    public double theta() {
        return theta(get(1) / get(0));
    }

    public int quadrant() {
        if (get(1) > 0) {
            if (get(0) > 0) {
                return 1;
            } else if (get(0) < 0) {
                return 2;
            }
        } else if (get(1) < 0) {
            if (get(0) < 0) {
                return 3;
            } else if (get(0) > 0) {
                return 4;
            }
        }
        return -1;
    }

    public void set(double magnitude, double theta) {
        set(0, magnitude);
        set(1, 0);
        rotateTo(theta);
    }

    //measured from the y-axis, vectors in the 4th quadrant are positive, vectors in the 3rd quadrant are negative
    public static double theta(double wp) {
        return -1 * Math.atan(1.0 / wp);
    }

}
