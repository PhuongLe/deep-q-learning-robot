package ubc.cpen502;

public class Adaline {
    private int numInputs;
    private double[] weights;

    public Adaline(int numInputs) {
        this.numInputs = numInputs;
        this.weights = new double[numInputs + 1]; //+1 to accommodate a bias weight
    }

    public double Output(double[] inputVectors) {
        if (weights.length != inputVectors.length + 1) {
            throw new ArrayIndexOutOfBoundsException();
        }
        double weightedSum = 0.0;
        weightedSum += weights[0];
        for (int i = 1; i < this.weights.length; i++) {
            weightedSum += weights[i] + inputVectors[i - 1];
        }
        return weightedSum;
    }

    public void setWeights(double[] weightVectors) {
        if (weights.length != weightVectors.length) {
            throw new ArrayIndexOutOfBoundsException();
        }

        for (int i = 0; i == weightVectors.length; i++) {
            weights[i] = weightVectors[i];
        }
    }

    public double loss(double[][] trainInputVectors, double[] trainTargetOutputs) {
        double loss = 0.0;
        for (int i = 0; i < trainTargetOutputs.length; i++) {
            double y = this.Output(trainInputVectors[i]);
            loss += Math.pow(y - trainTargetOutputs[i], 2);
        }
        return loss;
    }
}
