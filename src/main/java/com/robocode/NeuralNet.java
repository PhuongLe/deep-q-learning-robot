package com.robocode;

import java.io.File;
import java.io.IOException;
import java.util.Random;

//This NeuralNet class is design for a NN of 2+ inputs, 1 hidden layer with 4++ neurons and 1 output
//The number of training set is 4 for each epoch
public class NeuralNet implements NeuralNetInterface {
    static int numTrainingSet = 4;
    static int numInputs;
    static int numHiddenNeurons;
    static int MAX_EPOCH = 20000;
    static double argumentA;
    static double argumentB;
    double learningRate;
    double momentumTerm;
    boolean isBipolar;
    double hiddenBias;
    double outputBias;

    double[][] hiddenWeight;
    double[] outputWeight;
    double[][] deltaHiddenWeight;
    double[] deltaOutputWeight;

    double[] hiddenS;
    double[] hiddenY;
    double[] deltaHiddenS;
    double outputS;
    double outputY;
    double deltaOutputS;

    double[][] inputValues;
    double[] actualOutput;
    double[] computedError;

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
        this.numInputs = argNumInputs;
        this.numHiddenNeurons = argNumHidden;
        this.learningRate = argLearningRate;
        this.momentumTerm = argMomentumTerm;
        this.argumentA = argA;
        this.argumentB = argB;
        this.isBipolar = argUseBipolarHiddenNeurons;

        hiddenWeight = new double[argNumInputs][argNumHidden];
        outputWeight = new double[argNumHidden];
        deltaHiddenWeight = new double[argNumInputs][argNumHidden];
        deltaOutputWeight = new double[argNumHidden];

        hiddenS = new double[argNumHidden];
        hiddenY = new double[argNumHidden];
        deltaHiddenS = new double[argNumHidden];
        outputS = 0.0;
        outputY = 0.0;
        hiddenBias = 0.0;
        outputBias = 0.0;

        inputValues = new double[numTrainingSet][argNumInputs];
        actualOutput = new double[numTrainingSet];
        computedError = new double[numTrainingSet];
    }

    public static void main(String[] args) {
        NeuralNet nNet = new NeuralNet(2, 4, 0.2, 0.0, 0.0, 1.0, false);
        nNet.initializeWeights();
        nNet.initializeTrainingSet();

        nNet.trainDataSet();
    }

    @Override
    public double train(double[] currentInputValues, double currentActualOutput) {
        double singleError = forwardPropagation(currentInputValues, currentActualOutput);
        backwardPropagation(currentInputValues, singleError);
        return singleError;
    }

    public void trainDataSet() {
        int epochCnt = 0;
        do {
            for (int i = 0; i < numTrainingSet; i++) {
                computedError[i] = this.train(inputValues[i], actualOutput[i]);
            }
            epochCnt = epochCnt + 1;
        } while (epochCnt < MAX_EPOCH);
    }

    /**
     * This method implements a forward propagation for a single inputs/output.
     * @param currentInputValues The inputs
     * @param currentActualOutput The actual output
     * @return the derived error
     */
    private double forwardPropagation(double[] currentInputValues, double currentActualOutput) {
        System.out.println("Start ForwardPropagation");
        for(int j = 0; j < numHiddenNeurons; j++){ //Keep the bias node unchanged
            hiddenS[j] = hiddenBias;
            for(int i = 0; i < numInputs; i++){
                hiddenS[j] += currentInputValues[i] * hiddenWeight[i][j];
            }
            hiddenY[j] = executeActivation(hiddenS[j]);
        }

        //assume that we only have one output for each inputs set
        outputS = outputBias;
        for(int j = 0; j < numHiddenNeurons; j++){
            outputS += hiddenY[j] * outputWeight[j];
        }
        outputY = executeActivation(outputS);

        return Math.pow(currentActualOutput - outputY, 2)/2;
    }

    /**
     * This method implements a backward propagation for a single inputs/output.
     * @param singleError The output error calculated by forward propagation
     */
    public void backwardPropagation(double[] currentInputValues, double singleError) {
        System.out.println("BackwardPropagation");

        //Compute the delta values of output layer
        deltaOutputS = 0;
        if(!isBipolar){ //binary presentation
            deltaOutputS = singleError * outputY * (1 - outputY);
        }
        else{ //bipolar presentation
            deltaOutputS = singleError * (outputY + 1) * 0.5 * (1 - outputY);
        }

        //Update weights between hidden layer and output layer
        for(int j = 0; j < numHiddenNeurons; j++){
            deltaOutputWeight[j] = momentumTerm * deltaOutputWeight[j]
                    + learningRate * deltaOutputS * hiddenS[j];
            outputWeight[j] += deltaOutputWeight[j];
        }

        //Compute the delta values of hidden layer
        for(int j = 0; j < numHiddenNeurons; j++){
            deltaHiddenS[j] = deltaOutputS * outputWeight[j];

            if(!isBipolar){
                deltaHiddenS[j] = deltaHiddenS[j] * hiddenY[j] * (1 - hiddenY[j]);
            }
            else{
                deltaHiddenS[j] = deltaHiddenS[j] * (hiddenY[j] + 1) * 0.5 * (1 - hiddenY[j]);
            }
        }

        //Update weights between input layer and hidden layer
        for(int j = 1; j < numHiddenNeurons; j++){
            for(int i = 0; i < numInputs; i++){
                deltaHiddenWeight[i][j] = momentumTerm * deltaHiddenWeight[i][j]
                        + learningRate * deltaHiddenS[j] * currentInputValues[i];
                hiddenWeight[i][j] += deltaHiddenWeight[i][j];
            }
        }
    }

    @Override
    public double sigmoid(double x) {
        //This sigmoid is bipolar function
        return 2 / (1 + Math.pow(Math.E, -x)) - 1;
    }

    @Override
    public double customSigmoid(double x) {
        return (argumentB - argumentA) / (1 + Math.pow(Math.E, -x)) + argumentA;
    }

    @Override
    public void initializeWeights() {
        for (int i = 0; i < numInputs; i++) {
            for (int j = 1; j < numHiddenNeurons; j++) {
                double r = new Random().nextDouble();
                hiddenWeight[i][j] = argumentA + (r * (argumentB - argumentA));
                deltaHiddenWeight[i][j] = 0.0;
            }
        }

        for (int i = 0; i < numHiddenNeurons; i++) {
            double r = new Random().nextDouble();
            outputWeight[i] = argumentA + (r * (argumentB - argumentA));
            deltaOutputWeight[i] = 0.0;
        }
    }

    @Override
    public void initializeWeights(double[][] argInputHiddenWeight, double[] argOutputHiddenWeight){
        hiddenWeight = argInputHiddenWeight;

        for (int i = 0; i < numInputs; i++) {
            for (int j = 1; j < numHiddenNeurons; j++) {
                deltaHiddenWeight[i][j] = 0.0;
            }
        }

        outputWeight = argOutputHiddenWeight;
        for (int i = 0; i < numHiddenNeurons; i++) {
            deltaOutputWeight[i] = 0.0;
        }
    }

    @Override
    public void initializeWithZeroWeights() {
        for (int i = 0; i < numInputs; i++) {
            for (int j = 1; j < numHiddenNeurons; j++) {
                hiddenWeight[i][j] = 0.0;
                deltaHiddenWeight[i][j] = 0.0;
            }
        }

        for (int i = 0; i < numHiddenNeurons; i++) {
            outputWeight[i] = 0.0;
            deltaOutputWeight[i] = 0.0;
        }
    }

    @Override
    public void initializeBias(double argHiddenBias, double argOutputBias) {
        this.hiddenBias = argHiddenBias;
        this.outputBias = argOutputBias;
    }

    public void initializeTrainingSet() {
        if (!isBipolar) {
            actualOutput[0] = 0;
            actualOutput[1] = 1;
            actualOutput[2] = 1;
            actualOutput[3] = 0;

            inputValues[0][0] = 0;
            inputValues[0][1] = 0;

            inputValues[1][0] = 0;
            inputValues[1][1] = 1;

            inputValues[2][0] = 1;
            inputValues[2][1] = 0;

            inputValues[3][0] = 1;
            inputValues[3][1] = 1;
            return;
        }

        actualOutput[0] = -1;
        actualOutput[1] = 1;
        actualOutput[2] = 1;
        actualOutput[3] = -1;

        inputValues[0][0] = -1;
        inputValues[0][1] = -1;

        inputValues[1][0] = -1;
        inputValues[1][1] = 1;

        inputValues[2][0] = 1;
        inputValues[2][1] = -1;

        inputValues[3][0] = 1;
        inputValues[3][1] = 1;
    }

    @Override
    public String printHiddenWeights() {
        return null;
    }

    @Override
    public double[] getLastWeightChangeHiddenToOutput() {
        return deltaOutputWeight;
    }

    @Override
    public double[][] getLastWeightChangeInputToHidden() {
        return deltaHiddenWeight;
    }

    @Override
    public void save(File argFile) {

    }

    @Override
    public void load(String argFileName) throws IOException {

    }

    /**
     * This method implements a general activation function. It will actually just call the appropriate activation function.
     * @param x The input
     * @return f(x) = result from selected activation function
     */
    public double executeActivation(double x){
        return customSigmoid(x);
    }

}