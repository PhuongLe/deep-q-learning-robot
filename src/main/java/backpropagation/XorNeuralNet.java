package backpropagation;

import common.Activation;
import common.NeuralNetInterface;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

//This NeuralNet class is design for a NN of 2+ inputs, 1 hidden layer with 4++ neurons and 1 output
//The number of training set is 4 for each epoch
public class XorNeuralNet implements NeuralNetInterface {
    int numTrainingSet = 1;
    int numInputs;
    int numOutputs;
    int numHiddenNeurons;
    double argumentA;
    double argumentB;
    double learningRate;
    double momentumTerm;
    boolean isBipolar;

    double[][] hiddenWeight;
    double[][] outputWeight;
    double[][] deltaHiddenWeight;
    double[][] deltaOutputWeight;

    double[] hiddenS;
    double[] hiddenY;
    double[] deltaHiddenS;

    double[] outputS;
    double[] outputY;
    double[] deltaOutputS;

    double[][] inputValues;
    double[][] actualOutputs;

    Activation activationFunction;

    /**
     * Constructor. (Cannot be declared in an interface, but your implementation will need one)
     *
     * @param argNumInputs    The number of inputs in your input vector
     * @param argNumHiddenNeurons    The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
     * @param argLearningRate The learning rate coefficient
     * @param argMomentumTerm The momentum coefficient
     * @param argA            Integer lower bound of sigmoid used by the output neuron only.
     * @param argB            Integer upper bound of sigmoid used by the output neuron only.
     */
    public XorNeuralNet(
            int argNumInputs,
            int argNumHiddenNeurons,
            int argNumOutputs,
            double argLearningRate,
            double argMomentumTerm,
            double argA,
            double argB,
            boolean argUseBipolarHiddenNeurons,
            boolean initializeTrainingSet){
        this.numInputs = argNumInputs;
        this.numHiddenNeurons = argNumHiddenNeurons;
        this.numOutputs = argNumOutputs;
        this.learningRate = argLearningRate;
        this.momentumTerm = argMomentumTerm;
        this.argumentA = argA;
        this.argumentB = argB;
        this.isBipolar = argUseBipolarHiddenNeurons;

        hiddenWeight = new double[argNumInputs + 1][argNumHiddenNeurons];
        outputWeight = new double[argNumOutputs][argNumHiddenNeurons + 1];
        deltaHiddenWeight = new double[argNumInputs + 1][argNumHiddenNeurons];
        deltaOutputWeight = new double[argNumOutputs][argNumHiddenNeurons + 1];

        hiddenS = new double[argNumHiddenNeurons];
        hiddenY = new double[argNumHiddenNeurons];
        deltaHiddenS = new double[argNumHiddenNeurons];
        deltaOutputS = new double[argNumOutputs];
        outputS = new double[argNumOutputs];
        outputY = new double[argNumOutputs];

        this.activationFunction = new SigmoidActivation(this.argumentA, this.argumentB);
        if (initializeTrainingSet) {
            this.initializeTrainingSet();
        }
    }

    public XorNeuralNet(
            int argNumInputs,
            int argNumHidden,
            int argNumOutputs,
            double argLearningRate,
            double argMomentumTerm,
            double argA,
            double argB,
            boolean argUseBipolarHiddenNeurons){
        this(argNumInputs, argNumHidden, argNumOutputs, argLearningRate, argMomentumTerm, argA, argB, argUseBipolarHiddenNeurons, true);
    }

    @Override
    public void cloneWeights(NeuralNetInterface targetNetwork) {
        double[][] targetHiddenWeights = targetNetwork.getHiddenWeight();
        for (int i=0; i < numInputs + 1; i++){
            System.arraycopy(targetHiddenWeights[i], 0, this.hiddenWeight[i], 0, numHiddenNeurons);
        }

        double[][] targetOutputWeights = targetNetwork.getOutputWeights();
        for (int j=0; j < numOutputs; j++) {
            System.arraycopy(targetOutputWeights[j], 0, this.outputWeight[j], 0, numHiddenNeurons + 1);
        }
    }

    /**
     * This is to compute neural network's output by performing forward propagation
     * @param inputVector inputs vector
     * @return computed outputs vector of the given inputs vector
     */
    @Override
    public double[] outputFor(double[] inputVector) {
        for(int j = 0; j < numHiddenNeurons; j++){ //Keep the bias node unchanged
            hiddenS[j] = hiddenWeight[numInputs][j];
            for(int i = 0; i < numInputs; i++){
                hiddenS[j] += inputVector[i] * hiddenWeight[i][j];
            }
            hiddenY[j] = computeActivation(hiddenS[j]);
        }

        //assume that we only have one output for each inputs set
        for(int i = 0; i < numOutputs; i++) {
            outputS[i] = outputWeight[i][numHiddenNeurons];
            for (int j = 0; j < numHiddenNeurons; j++) {
                outputS[i] += hiddenY[j] * outputWeight[i][j];
            }
            outputY[i] = computeActivation(outputS[i]);
        }

        return outputY;
    }

    @Override
    public double train(double[] inputVector, double[] expectedOutput) {
        double totalRMSErrors = 0;
        double[] outputY = outputFor(inputVector);
        double[] errors = new double[numOutputs];
        for(int i = 0; i < numOutputs; i++) {
            double singleError = expectedOutput[i] - outputY[i];
            errors[i] = singleError;
            totalRMSErrors = 0.5*Math.pow(singleError, 2);
        }
        backwardPropagation(inputVector, errors);
        return totalRMSErrors/numOutputs;
    }

    /**
     * This method implements a backward propagation for a single inputs/output.
     * @param errors The output error calculated by forward propagation
     */
    @Override
    public void backwardPropagation(double[] inputVector, double[] errors) {
        //Compute the delta values of output layer
        for(int i = 0; i < numOutputs; i++) {
            deltaOutputS[i] = computeDerivativeOfActivation(errors[i], outputY[i]);
        }

        //Update weights between hidden layer and output layer
        for(int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < numHiddenNeurons; j++) {
                deltaOutputWeight[i][j] = momentumTerm * deltaOutputWeight[i][j]
                        + learningRate * deltaOutputS[i] * hiddenY[j];
                outputWeight[i][j] += deltaOutputWeight[i][j];
            }
            //for bias weights
            deltaOutputWeight[i][numHiddenNeurons] = momentumTerm * deltaOutputWeight[i][numHiddenNeurons]
                    + learningRate * deltaOutputS[i] * 1;
            outputWeight[i][numHiddenNeurons] += deltaOutputWeight[i][numHiddenNeurons];
        }

        //Compute the delta values of hidden layer
        for (int j = 0; j < numHiddenNeurons; j++) {
            double errorAtj = 0;
            for(int i = 0; i < numOutputs; i++) {
                errorAtj += deltaOutputS[i] * outputWeight[i][j];
            }
            deltaHiddenS[j] = computeDerivativeOfActivation(errorAtj, hiddenY[j]);
        }

        //Update weights between input layer and hidden layer
        for (int j = 0; j < numHiddenNeurons; j++) {
            for (int i = 0; i < numInputs; i++) {
                deltaHiddenWeight[i][j] = momentumTerm * deltaHiddenWeight[i][j]
                        + learningRate * deltaHiddenS[j] * inputVector[i];
                hiddenWeight[i][j] += deltaHiddenWeight[i][j];
            }
            //for bias' weights
            deltaHiddenWeight[numInputs][j] = momentumTerm * deltaHiddenWeight[numInputs][j]
                    + learningRate * deltaHiddenS[j] * 1;
            hiddenWeight[numInputs][j] += deltaHiddenWeight[numInputs][j];
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
        for (int i = 0; i < numInputs + 1; i++) {
            for (int j = 0; j < numHiddenNeurons; j++) {
                hiddenWeight[i][j] = generateRandomWeight();
                deltaHiddenWeight[i][j] = 0.0;
            }
        }

        for (int i = 0; i < numHiddenNeurons + 1; i++) {
            for (int j = 0; j < numOutputs; j++) {
                outputWeight[j][i] = generateRandomWeight();
                deltaOutputWeight[j][i] = 0.0;
            }
        }
    }

    private double generateRandomWeight(){
        return new Random().nextDouble() - 0.5;
    }

    @Override
    public void initializeWeights(double[][] argHiddenWeight, double[][] argOutputWeight){
        hiddenWeight = argHiddenWeight;

        for (int i = 0; i < numInputs + 1; i++) {
            for (int j = 1; j < numHiddenNeurons; j++) {
                deltaHiddenWeight[i][j] = 0.0;
            }
        }

        outputWeight = argOutputWeight;
        for (int i = 0; i < numHiddenNeurons + 1; i++) {
            for (int j = 0; j < numOutputs; j++) {
                deltaOutputWeight[j][i] = 0.0;
            }
        }
    }

    @Override
    public void initializeWeights(double[][] argHiddenWeight, double[][] argOutputWeight, double[][] argDeltaHiddenWeight, double[][] argDeltaOutputWeight){
        hiddenWeight = argHiddenWeight;
        deltaHiddenWeight = argDeltaHiddenWeight;
        outputWeight = argOutputWeight;
        deltaOutputWeight = argDeltaOutputWeight;
    }

    @Override
    public void initializeWithZeroWeights() {
        for (int i = 0; i < numInputs + 1; i++) {
            for (int j = 1; j < numHiddenNeurons; j++) {
                hiddenWeight[i][j] = 0.0;
                deltaHiddenWeight[i][j] = 0.0;
            }
        }

        for (int i = 0; i < numHiddenNeurons + 1; i++) {
            for (int j = 0; j < numOutputs; j++) {
                outputWeight[j][i] = 0.0;
                deltaOutputWeight[j][i] = 0.0;
            }
        }
    }

    @Override
    public void initializeTrainingSet(){
        numTrainingSet = 4;
        inputValues = new double[numTrainingSet][numInputs];
        actualOutputs = new double[numTrainingSet][numOutputs];

        if (!isBipolar) {
            actualOutputs[0][0] = 0;
            actualOutputs[1][0] = 1;
            actualOutputs[2][0] = 1;
            actualOutputs[3][0] = 0;

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
        actualOutputs[0][0] = -1;
        actualOutputs[1][0] = 1;
        actualOutputs[2][0] = 1;
        actualOutputs[3][0] = -1;

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
    public String printAllWeights() {
        StringBuilder str = new StringBuilder();
        str.append("\n");
        str.append("WeightToOutput = ");
        str.append("{");


        for (int i = 0; i < numHiddenNeurons + 1; i++) {
            for (int j = 0; j < numOutputs; j++) {
                str.append(" w" + i + "," + j + ": " + outputWeight[j][i] + " ");
            }
        }
        str.append("}");

        str.append("\n");
        str.append("WeightToHidden = ");

        for (int i = 0; i < numInputs + 1; i++) {
            str.append("{");
            for (int j = 1; j < numHiddenNeurons; j++) {
                str.append("w" + i +  "," + j + ":" + hiddenWeight[i][j] + " ");
            }
            str.append("}");
        }

        str.append("\n");
        return str.toString();
    }

    @Override
    public double[][] getLastWeightChangeToOutput() {
        return deltaOutputWeight;
    }

    @Override
    public double[][] getLastWeightChangeToHidden() {
        return deltaHiddenWeight;
    }

    @Override
    public double[][] getOutputWeights() {
        return outputWeight;
    }

    @Override
    public double[][] getHiddenWeight() {
        return hiddenWeight;
    }

    @Override
    public void save(File argFile) throws IOException {
        FileWriter writer;
        PrintWriter output;

        try{
            writer = new FileWriter(argFile, true);
            output = new PrintWriter(writer);
        } catch (IOException e) {
            System.out.println("*** Could not create output stream for NN save file");
            return;
        }

        output.println(numInputs);
        output.println(numHiddenNeurons);

        //First save the weights from the input to hidden neurons (one line per weight)
        for (int i=0; i < numInputs + 1; i++){
            for (int j=0; j < numHiddenNeurons; j++){
                output.println(hiddenWeight[i][j]);
            }
        }

        //Now save the weights from hidden to the output neuron
        for (int i=0; i < numHiddenNeurons + 1; i++){
            for (int j=0; j < numOutputs; j++) {
                output.println(outputWeight[j][i]);
            }
        }

        output.close();
        writer.close();
    }

    /**
     * Loads the LUT or neural net weights from file. The load must of course
     * have knowledge of how the data was written out by the save method.
     * You should raise an error in the case that an attempt is being
     * made to load data into an LUT or neural net whose structure does not match
     * the data in the file. (e.g. wrong number of hidden neurons).
     * @throws IOException ioexception
     */
    @Override
    public void load(String argFileName) throws IOException {
        FileInputStream inputFile = new FileInputStream(argFileName);
        BufferedReader inputReader = new BufferedReader(new InputStreamReader(inputFile));

        // Check that NN defined for file matches that created
        int numInputInFile = Integer.parseInt(inputReader.readLine());
        int numHiddenInFile = Integer.parseInt(inputReader.readLine());
        if(numInputInFile != numInputs){
            System.out.println("--- Number of inputs in file is " + numInputInFile + "Expected " + numInputs);
            inputReader.close();
            throw new IOException();
        }
        if(numHiddenInFile != numHiddenNeurons){
            System.out.println("--- Number of hidden in file is " + numHiddenInFile + "Expected " + numHiddenNeurons);
            inputReader.close();
            throw new IOException();
        }
        // Load the weights from input layer to hidden neurons (one line per weight)
        // Loads the weights for the bias as well
        for (int i = 0; i < numInputs + 1; i++){
            for (int j = 0; j < numHiddenNeurons; j++){
                hiddenWeight[i][j] = Double.parseDouble(inputReader.readLine());
            }
        }

        // Load the weights from the hidden layer to the output
        // Loads the weight for the bias as well
        for (int i = 0; i < numHiddenNeurons + 1; i++){
            for (int j=0; j < numOutputs; j++) {
                outputWeight[j][i] = Double.parseDouble(inputReader.readLine());
            }
        }

        // Close file
        inputFile.close();
        inputReader.close();
    }

    /**
     * This method implements a general activation function. It will actually just call the appropriate activation function.
     * @param x The input
     * @return f(x) = result from selected activation function
     */
    public double computeActivation(double x){
        return customSigmoid(x);
    }

    @Override
    public void setActivation(Activation argActivation) {
        this.activationFunction = argActivation;
    }


    @Override
    public int run(String outputFileName, double target, boolean showErrorAtEachEpoch, boolean showHiddenWeightsAtEachEpoch, boolean showErrorAtConverge) throws IOException {
        double error;
        List<Double> errors = new ArrayList<>();

        int epochsToReachTarget = 0;
        boolean targetReached = false;

        String initializedWeights = this.printAllWeights();

        int epochCnt = 0;
        do {
            error = 0.0;
            for (int i = 0; i < numTrainingSet; i++) {
                double computedError = this.train(this.inputValues[i], this.actualOutputs[i]);
                //error += 0.5*Math.pow(computedError,2);
                error += computedError;
            }
            errors.add(error);
            if (showErrorAtEachEpoch) System.out.println("--+ Error at epoch " + epochCnt + " is " + error);
            if (showHiddenWeightsAtEachEpoch) System.out.println("--+ Hidden weights at epoch " + epochCnt + " " + this.printAllWeights());

            if (error < target){
                if (showErrorAtConverge) {
                    System.out.println("Yo!! Error = " + error + " after " + epochCnt + " epochs");
                    System.out.println(initializedWeights);
                }
                //output.println("Yo!! Error = " + error + " after " + epochCnt + " epochs");
                saveRunResult(outputFileName, errors);
                epochsToReachTarget = epochCnt;
                targetReached = true;
                break;
            }

            epochCnt = epochCnt + 1;
        } while (epochCnt < MAX_EPOCH);

        if (targetReached){
            System.out.println("--+ Target error reached at " + epochsToReachTarget+" epochs");
            return epochCnt;
        }
        else {
            System.out.println("-** Target not reached");
            return DID_NOT_CONVERGE;
        }
    }

    protected void saveRunResult(String outputFileName, List<Double> errors) throws IOException {
        File file = new File(outputFileName);
        FileWriter writer = new FileWriter(file, true);
        PrintWriter output = new PrintWriter(writer);

        int epochCnt = 0;
        StringBuilder epochIndexes = new StringBuilder();
        StringBuilder errorString = new StringBuilder();
        for (Double err : errors) {
            epochIndexes.append(epochCnt).append(",");
            errorString.append(err).append(",");
            epochCnt ++;
        }
        epochIndexes = new StringBuilder(epochIndexes.substring(0, epochIndexes.length() - 1));
        errorString = new StringBuilder(errorString.substring(0, errorString.length() - 1));
        output.print(epochIndexes);
        output.println();
        output.print(errorString);
        output.println();
        output.println();

        output.close();
        writer.close();
    }
}
