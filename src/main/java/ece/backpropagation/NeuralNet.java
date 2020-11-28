package ece.backpropagation;

import ece.common.Activation;
import ece.common.NeuralNetInterface;
import robocode.RobocodeFileOutputStream;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;

//This NeuralNet class is design for a NN of 2+ inputs, 1 hidden layer with 4++ neurons and 1 output
//The number of training set is 4 for each epoch
public class NeuralNet implements NeuralNetInterface {
    static int numTrainingSet = 4;
    static int numInputs;
    static int numHiddenNeurons;
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

    boolean onlineUpdatePerRound = true;

    Activation activationFunction;

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

        this.activationFunction = new SigmoidActivation(this.argumentA, this.argumentB);
    }

    @Override
    public double outputFor(double[] X) {
        return 0;
    }

    @Override
    public double train(double[] currentInputValues, double currentActualOutput) {
        double singleError = forwardPropagation(currentInputValues, currentActualOutput);
        backwardPropagation(currentInputValues, singleError);
        return singleError;
    }

    /**
     * This method implements a forward propagation for a single inputs/output.
     * @param currentInputValues The inputs
     * @param currentActualOutput The actual output
     * @return the derived error
     */
    private double forwardPropagation(double[] currentInputValues, double currentActualOutput) {
        //System.out.println("Start ForwardPropagation");
        for(int j = 0; j < numHiddenNeurons; j++){ //Keep the bias node unchanged
            hiddenS[j] = hiddenBias;
            for(int i = 0; i < numInputs; i++){
                hiddenS[j] += currentInputValues[i] * hiddenWeight[i][j];
            }
            hiddenY[j] = computeActivation(hiddenS[j]);
        }

        //assume that we only have one output for each inputs set
        outputS = outputBias;
        for(int j = 0; j < numHiddenNeurons; j++){
            outputS += hiddenY[j] * outputWeight[j];
        }
        outputY = computeActivation(outputS);

        return currentActualOutput - outputY;
    }

    /**
     * This method implements a backward propagation for a single inputs/output.
     * @param singleError The output error calculated by forward propagation
     */
    public void backwardPropagation(double[] currentInputValues, double singleError) {
        if (onlineUpdatePerRound) {
            //Compute the delta values of output layer
            deltaOutputS = computeDerivativeOfActivation(singleError, outputY);

            //Update weights between hidden layer and output layer
            for (int j = 0; j < numHiddenNeurons; j++) {
                deltaOutputWeight[j] = momentumTerm * deltaOutputWeight[j]
                        + learningRate * deltaOutputS * hiddenY[j];
                outputWeight[j] += deltaOutputWeight[j];
            }

            //Compute the delta values of hidden layer
            for (int j = 0; j < numHiddenNeurons; j++) {
                double errorAtj = deltaOutputS * outputWeight[j];
                deltaHiddenS[j] = computeDerivativeOfActivation(errorAtj, hiddenY[j]);
            }

            //Update weights between input layer and hidden layer
            for (int j = 0; j < numHiddenNeurons; j++) {
                for (int i = 0; i < numInputs; i++) {
                    deltaHiddenWeight[i][j] = momentumTerm * deltaHiddenWeight[i][j]
                            + learningRate * deltaHiddenS[j] * currentInputValues[i];
                    hiddenWeight[i][j] += deltaHiddenWeight[i][j];
                }
            }
            return;
        }
        //Compute the delta values of output layer
        deltaOutputS = computeDerivativeOfActivation(singleError, outputY);

        //Compute the delta values of hidden layer
        for (int j = 0; j < numHiddenNeurons; j++) {
            double errorAtj = deltaOutputS * outputWeight[j];
            deltaHiddenS[j] = computeDerivativeOfActivation(errorAtj, hiddenY[j]);
        }

        //Update weights between hidden layer and output layer
        for (int j = 0; j < numHiddenNeurons; j++) {
            deltaOutputWeight[j] = momentumTerm * deltaOutputWeight[j]
                    + learningRate * deltaOutputS * hiddenY[j];
            outputWeight[j] += deltaOutputWeight[j];
        }

        //Update weights between input layer and hidden layer
        for (int j = 0; j < numHiddenNeurons; j++) {
            for (int i = 0; i < numInputs; i++) {
                deltaHiddenWeight[i][j] = momentumTerm * deltaHiddenWeight[i][j]
                        + learningRate * deltaHiddenS[j] * currentInputValues[i];
                hiddenWeight[i][j] += deltaHiddenWeight[i][j];
            }
        }
    }

    private double computeDerivativeOfActivation(double error, double y) {
        return error*activationFunction.ComputeDerivative(y);
    }

    @Override
    public double customSigmoid(double x) {
        return activationFunction.ComputeY(x);
    }

    @Override
    public void initializeWeights() {
        for (int i = 0; i < numInputs; i++) {
            for (int j = 0; j < numHiddenNeurons; j++) {
                hiddenWeight[i][j] = generateRandomWeight();
                deltaHiddenWeight[i][j] = 0.0;
            }
        }

        for (int i = 0; i < numHiddenNeurons; i++) {
            outputWeight[i]  = generateRandomWeight();

            deltaOutputWeight[i] = 0.0;
        }

        this.hiddenBias = generateRandomWeight();
        this.outputBias = generateRandomWeight();
    }

    private double generateRandomWeight(){
        return new Random().nextDouble() - 0.5;
    }

    @Override
    public void initializeWeights(double[][] argHiddenWeight, double[] argOutputWeight){
        hiddenWeight = argHiddenWeight;

        for (int i = 0; i < numInputs; i++) {
            for (int j = 1; j < numHiddenNeurons; j++) {
                deltaHiddenWeight[i][j] = 0.0;
            }
        }

        outputWeight = argOutputWeight;
        for (int i = 0; i < numHiddenNeurons; i++) {
            deltaOutputWeight[i] = 0.0;
        }
    }

    @Override
    public void initializeWeights(double[][] argHiddenWeight, double[] argOutputWeight, double[][] argDeltaHiddenWeight, double[] argDeltaOutputWeight){
        hiddenWeight = argHiddenWeight;
        deltaHiddenWeight = argDeltaHiddenWeight;
        outputWeight = argOutputWeight;
        deltaOutputWeight = argDeltaOutputWeight;
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
        StringBuilder str = new StringBuilder();
        str.append("\n");
        str.append("WeightToOutput = ");
        str.append("{");
        for (int i=0; i<outputWeight.length; i ++ ){
            str.append(" w" + i + ": "+outputWeight[i] + " ");
        }
        str.append("}");

        str.append("\n");
        str.append("WeightToHidden = ");
        for (int i=0; i<hiddenWeight.length; i ++ ){
            str.append("{");
            for (int j=0; j<hiddenWeight[i].length; j ++ ) {
                str.append("w" + i + j + ":" + hiddenWeight[i][j] + " ");
            }
            str.append("}");
        }
        str.append("\n");
        return str.toString();
    }

    @Override
    public double[] getLastWeightChangeToOutput() {
        return deltaOutputWeight;
    }

    @Override
    public double[][] getLastWeightChangeToHidden() {
        return deltaHiddenWeight;
    }

    @Override
    public double[] getLastWeightToOutput() {
        return outputWeight;
    }

    @Override
    public double[][] getLastWeightToHidden() {
        return hiddenWeight;
    }

    @Override
    public void save(File argFile) {
        PrintStream saveFile = null;

        try{
            saveFile = new PrintStream(new RobocodeFileOutputStream(argFile));
        } catch (IOException e) {
            System.out.println("*** Could not create output stream for NN save file");
        }

        saveFile.println(numInputs);
        saveFile.println(numHiddenNeurons);

        //First save the weights from the input to hidden neurons (one line per weight)
        for (int i=0; i<numHiddenNeurons; i++){
            for (int j=0; j < numInputs;j++){
                saveFile.println(hiddenWeight[i][j]);
            }
            //saveFile.println(hiddenWeight[i][numInputs]);//todo save bias weight for this hidden neuron too
            saveFile.println(hiddenBias);
        }

        //Now save the weights from hidden to the output neuron
        for (int i=0; i<numHiddenNeurons; i++){
            saveFile.println(outputWeight[i]);
        }
        saveFile.println(outputBias); //save bias weight for output neuron too
        //saveFile.println(weightHiddenToOutput[numHidden]

        saveFile.close();
    }

    @Override
    public void load(String argFileName) throws IOException {

    }

    /**
     * This method implements a general activation function. It will actually just call the appropriate activation function.
     * @param x The input
     * @return f(x) = result from selected activation function
     */
    public double computeActivation(double x){
        return customSigmoid(x);
    }

    /**
     * This method is to enable batch update weights for each propagation round only
     */
    @Override
    public void enableBatchUpdateOption(){
        this.onlineUpdatePerRound = false;
    }

    @Override
    public void setActivation(Activation argActivation) {
        this.activationFunction = argActivation;
    }

}
