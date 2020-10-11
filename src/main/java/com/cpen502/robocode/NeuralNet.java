package com.cpen502.robocode;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class NeuralNet implements NeuralNetInterface {

    static int outputVal = 1;
    static int biasVal = 1;
    static int numTrainingSet = 4;
    static int inputVal;
    static int hiddenVal;
    static int MAX_EPOCH = 20000;
    static double argumentA;
    static double argumentB;
    double learningRate;
    double momentumRate;
    boolean isBipolar;

    double[][] inputHiddenWeight;
    double[][] outputHiddenWeight;
    double[][] inputDeltaHiddenWeight;
    double[][] outputDeltaHiddenWeight;

    double[][] inputValues;
    double[][] actualOutput;

    /**
     * Constructor. (Cannot be declared in an interface, but your implementation will need one)
     *
     * @param argNumInputs    The number of inputs in your input vector
     * @param argNumHidden    The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
     * @param argLearningRate The learning rate coefficient
     * @param argMomentumTerm The momentum coefficient
     * @param argA            Integer lower bound of sigmoid used by the output neuron only.
     * @param argB            Integer upper bound of sigmoid used by the output neuron only.
     */
    public NeuralNet(
            int argNumInputs,
            int argNumHidden,
            double argLearningRate,
            double argMomentumTerm,
            double argA,
            double argB,
            boolean argUseBipolarHiddenNeurons) {
        this.inputVal = argNumInputs;
        this.hiddenVal = argNumHidden;
        this.learningRate = argLearningRate;
        this.momentumRate = argMomentumTerm;
        this.argumentA = argA;
        this.argumentB = argB;
        this.isBipolar = argUseBipolarHiddenNeurons;

        inputHiddenWeight = new double[argNumInputs][argNumHidden];
        outputHiddenWeight = new double[argNumHidden][outputVal];
        inputDeltaHiddenWeight = new double[argNumInputs][argNumHidden];
        outputDeltaHiddenWeight = new double[argNumHidden][outputVal];
        inputValues = new double[numTrainingSet][argNumInputs];
        actualOutput = new double[numTrainingSet][outputVal];
    }

    public static void main(String[] args) {
        NeuralNet nNet = new NeuralNet(3, 4, 0.2, 0.0, -0.5, 0.5, false);
        nNet.initializeWeights();
        nNet.initializeTrainingSet();
        System.out.println("done");

        nNet.trainDataSet();
    }

    public void trainDataSet() {
        int epochCnt = 0;
        do {

            for (int i = 0; i < numTrainingSet; i++) {
                forwardFeed();
                backPropagation();
            }
            epochCnt = epochCnt + 1;
        } while (epochCnt < MAX_EPOCH);
    }

    public void forwardFeed() {
        System.out.println("ForwardFeed");
    }

    public void backPropagation() {
        System.out.println("Backpropagate");
    }

    @Override
    public double sigmoid(double x) {
        return 0;
    }

    @Override
    public double customSigmoid(double x) {
        return 0;
    }

    @Override
    public void initializeWeights() {
        for (int i = 0; i < inputVal; i++) {
            for (int j = 1; j < hiddenVal; j++) {
                double r = new Random().nextDouble();
                inputHiddenWeight[i][j] = argumentA + (r * (argumentB - argumentA));
                inputDeltaHiddenWeight[i][j] = 0.0;
            }
        }

        for (int i = 0; i < hiddenVal; i++) {
            for (int j = 0; j < outputVal; j++) {
                double r = new Random().nextDouble();
                outputHiddenWeight[i][j] = argumentA + (r * (argumentB - argumentA));
                outputDeltaHiddenWeight[i][j] = 0.0;
            }
        }
    }

    public void initializeTrainingSet() {
        if (!isBipolar) {
            actualOutput[0][0] = 0;
            actualOutput[1][0] = 1;
            actualOutput[2][0] = 1;
            actualOutput[3][0] = 0;

            inputValues[0][0] = biasVal;
            inputValues[0][1] = 0;
            inputValues[0][2] = 0;

            inputValues[1][0] = biasVal;
            inputValues[1][1] = 0;
            inputValues[1][2] = 1;

            inputValues[2][0] = biasVal;
            inputValues[2][1] = 1;
            inputValues[2][2] = 0;

            inputValues[3][0] = biasVal;
            inputValues[3][1] = 1;
            inputValues[3][2] = 1;

        } else {
            actualOutput[0][0] = -1;
            actualOutput[1][0] = 1;
            actualOutput[2][0] = 1;
            actualOutput[3][0] = -1;

            inputValues[0][0] = biasVal;
            inputValues[0][1] = -1;
            inputValues[0][2] = -1;

            inputValues[1][0] = biasVal;
            inputValues[1][1] = -1;
            inputValues[1][2] = 1;

            inputValues[2][0] = biasVal;
            inputValues[2][1] = 1;
            inputValues[2][2] = -1;

            inputValues[3][0] = biasVal;
            inputValues[3][1] = 1;
            inputValues[3][2] = 1;

        }
    }

    @Override
    public void zeroWeights() {

    }

    @Override
    public double outputFor(double[] X) {
        return 0;
    }

    @Override
    public double train(double[] X, double argValue) {
        return 0;
    }

    @Override
    public void save(File argFile) {

    }

    @Override
    public void load(String argFileName) throws IOException {

    }
}
