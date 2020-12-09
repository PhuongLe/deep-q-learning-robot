package ece.backpropagation;

import ece.common.Activation;
import ece.common.NeuralNetInterface;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

//This NeuralNet class is design for a NN of 2+ inputs, 1 hidden layer with 4++ neurons and 1 output
//The number of training set is 4 for each epoch
public class XorNeuralNet implements NeuralNetInterface {
    static String baseFolder = "d:\\Google Drive\\LXP\\UBC\\Term 3\\CPEN 502 - ML\\Assignments\\Robocode\\out\\report\\" + new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss") .format(new Date());
    static String logFileName = baseFolder+ "-neural-net.log";
    int numTrainingSet = 4;
    int numInputs;
    int numHiddenNeurons;
    double argumentA;
    double argumentB;
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
    public XorNeuralNet(
            int argNumInputs,
            int argNumHidden,
            double argLearningRate,
            double argMomentumTerm,
            double argA,
            double argB,
            boolean argUseBipolarHiddenNeurons,
            boolean initializeTrainingSet){
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
        hiddenBias = 1.0;
        outputBias = 1.0;

        this.activationFunction = new SigmoidActivation(this.argumentA, this.argumentB);
        if (initializeTrainingSet) {
            this.initializeTrainingSet();
        }
    }

    public XorNeuralNet(
            int argNumInputs,
            int argNumHidden,
            double argLearningRate,
            double argMomentumTerm,
            double argA,
            double argB,
            boolean argUseBipolarHiddenNeurons){
        this(argNumInputs, argNumHidden, argLearningRate, argMomentumTerm, argA, argB, argUseBipolarHiddenNeurons, true);
    }

    @Override
    public void cloneWeights(NeuralNetInterface targetNetwork) throws IOException {
        double[][] targetHiddentWeights = targetNetwork.getHiddenWeight();
        for (int i=0; i<numInputs; i++){
            for (int j=0; j < numHiddenNeurons;j++){
                this.hiddenWeight[i][j] = targetHiddentWeights[i][j];
            }
        }

        double[] targetOutputWeights = targetNetwork.getOutputWeights();
        for (int i=0; i<numHiddenNeurons; i++){
            this.outputWeight[i] = targetOutputWeights[i];
        }
    }

    /**
     * This is to compute neural network's output by performing forward propagation
     * @param inputVector
     * @return
     */
    @Override
    public double outputFor(double[] inputVector) {
        //System.out.println("Start ForwardPropagation");
        for(int j = 0; j < numHiddenNeurons; j++){ //Keep the bias node unchanged
            hiddenS[j] = hiddenBias;
            for(int i = 0; i < numInputs; i++){
                hiddenS[j] += inputVector[i] * hiddenWeight[i][j];
            }
            hiddenY[j] = computeActivation(hiddenS[j]);
        }

        //assume that we only have one output for each inputs set
        outputS = outputBias;
        for(int j = 0; j < numHiddenNeurons; j++){
            outputS += hiddenY[j] * outputWeight[j];
        }
        outputY = computeActivation(outputS);

        return outputY;    }

    @Override
    public double train(double[] inputVector, double expectedOutput) {
        double outputY = outputFor(inputVector);
        double singleError = expectedOutput - outputY;
        backwardPropagation(inputVector, singleError);
        return singleError;
    }

    /**
     * This method implements a backward propagation for a single inputs/output.
     * @param singleError The output error calculated by forward propagation
     */
    @Override
    public void backwardPropagation(double[] inputVector, double singleError) {
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
                            + learningRate * deltaHiddenS[j] * inputVector[i];
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
                        + learningRate * deltaHiddenS[j] * inputVector[i];
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

    @Override
    public void initializeTrainingSet(){
        inputValues = new double[numTrainingSet][numInputs];
        actualOutput = new double[numTrainingSet];

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
    public String printAllWeights() {
        StringBuilder str = new StringBuilder();
        str.append("\n");
        str.append("WeightToOutput = ");
        str.append("{");
        for (int i=0; i<outputWeight.length; i ++ ){
            str.append(" w" + i + ": " + outputWeight[i] + " ");
        }
        str.append("}");

        str.append("\n");
        str.append("WeightToHidden = ");
        for (int i=0; i< hiddenWeight.length; i ++ ){
            str.append("{");
            for (int j=0; j< hiddenWeight[i].length; j ++ ) {
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
    public double[] getOutputWeights() {
        return outputWeight;
    }

    @Override
    public double[][] getHiddenWeight() {
        return hiddenWeight;
    }

    @Override
    public double getHiddenBias() {
        return hiddenBias;
    }

    @Override
    public double getOutputBias() {
        return outputBias;
    }

    @Override
    public int getNumInputs() {
        return numInputs;
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
        output.println(hiddenBias);
        output.println(outputBias); //save bias weight for output neuron too

        //First save the weights from the input to hidden neurons (one line per weight)
        for (int i=0; i < numInputs; i++){
            for (int j=0; j < numHiddenNeurons; j++){
                output.println(hiddenWeight[i][j]);
            }
        }

        //Now save the weights from hidden to the output neuron
        for (int i=0; i < numHiddenNeurons; i++){
            output.println(outputWeight[i]);
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
     * @throws IOException
     */
    // Source: Dr. Sarkaria's code from tutorial class
    @Override
    public void load(String argFileName) throws IOException {
        FileInputStream inputFile = new FileInputStream(argFileName);
        BufferedReader inputReader = new BufferedReader(new InputStreamReader(inputFile));

        // Check that NN defined for file matches that created
        int numInputInFile = Integer.valueOf(inputReader.readLine());
        int numHiddenInFile = Integer.valueOf(inputReader.readLine());
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
        hiddenBias = Double.valueOf(inputReader.readLine());
        outputBias = Double.valueOf(inputReader.readLine());

        // Load the weights from input layer to hidden neurons (one line per weight)
        // Loads the weights for the bias as well
        for (int i = 0; i < numInputs; i++){
            for (int j = 0; j < numHiddenNeurons; j++){
                hiddenWeight[i][j] = Double.valueOf(inputReader.readLine());
            }
        }

        // Load the weights from the hidden layer to the output
        // Loads the weight for the bias as well
        for (int i = 0; i < numHiddenNeurons; i++){
            outputWeight[i] = Double.valueOf(inputReader.readLine());
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

    /**
     * This method is to enable batch update weights for each propagation round only
     */
    public void enableBatchUpdateOption(){
        this.onlineUpdatePerRound = false;
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
                double computedError = this.train(this.inputValues[i], this.actualOutput[i]);
                error += 0.5*Math.pow(computedError,2);
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
        String epochIndexes = "";
        String errorString = "";
        for (Double err : errors) {
            epochIndexes += epochCnt + ",";
            errorString += err + ",";
            epochCnt ++;
        }
        epochIndexes = epochIndexes.substring(0, epochIndexes.length() - 1);
        errorString = errorString.substring(0, errorString.length() - 1);
        output.print(epochIndexes);
        output.println();
        output.print(errorString);
        output.println();
        output.println();

        output.close();
        writer.close();
    }
}
