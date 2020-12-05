package ece.backpropagation;

import ece.robocode.LogFile;
import javafx.util.Pair;
import robocode.RobocodeFileOutputStream;
import ece.common.NeuralNetInterfaceSS;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class NeuralNetED implements NeuralNetInterfaceSS{

    // Files
    static LogFile log = null;
    static String baseFolder = "D:\\Work\\Courses\\Winter2020Term1\\CPEN502ArchitectureForLearningSystems\\Assignment3\\Phoebes_repo\\Logs_weights\\" + new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date());
    static String NNlogFileName = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date()) + "-NN-hyperparameters.log";
    static String weightsFileName = baseFolder + "-NN-weights.log";

    final double[] bias_arr = {1.0}; // The input for each neurons bias weight
    final double error_threshold = 0.05;

    int numOutput;
    int numInputs;
    int numHiddenLayerNeurons;
    double learningRate;
    double momentumValue;
    double sigmoidLB;
    double sigmoidUB;

    double[][] v_inputToHidden;
    double[][] w_hiddenToOutput;


//    double[][] xorPatterns = {{0,0}, {0,1}, {1,0}, {1,1}};
//    double[] xorExpectedOutput = {0,1,1,0};
//    double[][] xorPatterns = {{-1,-1}, {-1,1}, {1,-1}, {1,1}};
//    double[] xorExpectedOutput = {-1,1,1,-1};
//    int numPairs = xorExpectedOutput.length;
    double[][] xorPatterns;
    double[] xorExpectedOutput;
    int numPairs;


    double[] zDotProduct;
    double[] zActivation;

    //Compute dot product and activation at each output layer neuron
    double[] yDotProduct;
    double[] yActivation;

    double[][] v_inputToHidden_corrTermPrev;
    double[][] w_hiddenToOutput_corrTermPrev;

    // Constructor
    public NeuralNetED(int argNumOutput,
                       int argNumInputs,
                       int argNumHidden,
                       float argLearningRate,
                       float argMomentumTerm,
                       double argA,
                       double argB) throws IOException {

        loadProcessedLuT();

        this.numOutput = argNumOutput;
        this.numInputs = argNumInputs;
        this.numHiddenLayerNeurons = argNumHidden;
        this.learningRate = argLearningRate;
        this.momentumValue = argMomentumTerm;
        this.sigmoidLB = argA;
        this.sigmoidUB = argB;

        this.NN_ErrorBackpropagation();

    }

    private void NN_ErrorBackpropagation(){

        int num_attempts = 10;
        int max_iteration = 1000;
        int[] maxIterationEachAttempt = new int[num_attempts];

        double[][] track_mse_and_max_attempt = new double[num_attempts][max_iteration+1];
        for(int attemptInd = 0; attemptInd < num_attempts; attemptInd++){

            double[] rootMeanSquaredError = new double[max_iteration];
            int max_used_iteration = 0;

            // Initialize the weights
            // Initialize the weight correction terms (changes) to zero (no correction term for the first iteration)
            initializeWeights();

            // Declare some class variables
            this.zDotProduct = new double[this.numHiddenLayerNeurons];
            this.zActivation = new double[this.numHiddenLayerNeurons];
            this.yDotProduct = new double[this.numOutput];
            this.yActivation = new double[this.numOutput];

            // Loop through each epoch
            for(int iteration_index = 0; iteration_index < max_iteration; iteration_index++){

                double[] forward_prop_output = new double[this.xorPatterns.length];

                // Loop through each input vector
                // calculate MSE
                for(int xorInd = 0; xorInd < this.xorPatterns.length; xorInd++){

                    double[] xor_input = this.xorPatterns[xorInd];
                    double correct_y = this.xorExpectedOutput[xorInd];

//                System.out.println("v_inputToHidden");
//                print2d(bck_ret.vWeights);
//                System.out.println("w_hiddenToOutput");
//                print2d(bck_ret.wWeights);
//                System.out.println("--- Iteration num "+xor_index+" is complete ---");

                    // Call train() to return the MSE for each input vector
                    rootMeanSquaredError[iteration_index] += train(xor_input, correct_y);

                }

                rootMeanSquaredError[iteration_index] = Math.sqrt( (1.0/numPairs) * rootMeanSquaredError[iteration_index] );

                max_used_iteration = iteration_index;
                if(rootMeanSquaredError[iteration_index] < this.error_threshold){
                    System.out.println("Threshold reached. Breaking the loop. Iteration count: "+iteration_index+". MSE:" + rootMeanSquaredError[iteration_index] + " Iteration index:"+iteration_index);
                    break;
                }

            }

            maxIterationEachAttempt[attemptInd] = max_used_iteration;

            double[] max_used_tr_arr = {max_used_iteration};
            double[] rootMeanSquaredErrorFinal = concatenate(max_used_tr_arr, rootMeanSquaredError);
            track_mse_and_max_attempt[attemptInd] = rootMeanSquaredErrorFinal;
      }

        // Writing to file
        try{
            String output_str = prepare_2d_arr_for_matlab_import(track_mse_and_max_attempt);
            saveMSEArray("TrackMSEMAXiterationBipolarWithMomentum.txt", output_str);
        }
        catch (Exception e){
            System.out.println(e.getMessage());
        }
        try{
            // Save hyperparameters
            File log_file = new File("D:\\Work\\Courses\\Winter2020Term1\\CPEN502ArchitectureForLearningSystems\\Assignment3\\Phoebes_repo\\Logs_weights\\", NNlogFileName);
            if(!log_file.exists()){
                log_file.createNewFile();
            }
            LogFile log = new LogFile(log_file);
            log.stream.printf("numberInputs,   %d\n", numInputs);
            log.stream.printf("numberHiddenNeurons,   %d\n", numHiddenLayerNeurons);
            log.stream.printf("numberOutputs, %d\n", numOutput);
            log.stream.printf("learningRate, %2.6f\n", learningRate);
            log.stream.printf("momentum, %2.6f\n", momentumValue);
            log.closeStream();

            // Save weights
            File logWeights = new File(weightsFileName);
            if(!logWeights.exists()){
                logWeights.createNewFile();
            }
            save(logWeights);
        }
        catch (Exception e){
            System.out.println(e.getMessage());
        }


    }

    private double[] forward_propagation(double[] xorInputWithBias){

        //Compute dot product and activation at each hidden layer neuron
        for (int j = 0; j < this.numHiddenLayerNeurons; j++){
            this.zDotProduct[j] = dotProduct(xorInputWithBias, get_col_from_2d_arr(this.v_inputToHidden, j));
            this.zActivation[j] = customSigmoidWithInputBounds(this.zDotProduct[j], this.sigmoidLB, this.sigmoidUB);
        }

        //Concatenate bias to the hidden layer output
        double[] zOutputWithBias = concatenate(zActivation, this.bias_arr);

        //Compute dot product and activation at each output layer neuron
        for (int k = 0; k < this.numOutput; k++){
            this.yDotProduct[k] = dotProduct(zOutputWithBias, get_col_from_2d_arr(this.w_hiddenToOutput, k));
            this.yActivation[k] = customSigmoidWithInputBounds(this.yDotProduct[k], this.sigmoidLB, this.sigmoidUB);
        }

        return yActivation;
    }

    /**
     * @param x The input vector. An array of doubles.
     * @return The value returned by the LUT or NN for this input vector
     */
    public double outputFor(double [] x){
        // Concatenate input array with the bias term
        double[] xorInputWithBias = concatenate(x, this.bias_arr);

        //Compute dot product and activation at each hidden layer neuron
        for (int j = 0; j < this.numHiddenLayerNeurons; j++){
            this.zDotProduct[j] = dotProduct(xorInputWithBias, get_col_from_2d_arr(this.v_inputToHidden, j));
            this.zActivation[j] = customSigmoidWithInputBounds(this.zDotProduct[j], this.sigmoidLB, this.sigmoidUB);
        }

        //Concatenate bias to the hidden layer output
        double[] zOutputWithBias = concatenate(zActivation, this.bias_arr);

        //Compute dot product and activation at each output layer neuron
        for (int k = 0; k < this.numOutput; k++){
            this.yDotProduct[k] = dotProduct(zOutputWithBias, get_col_from_2d_arr(this.w_hiddenToOutput, k));
            this.yActivation[k] = customSigmoidWithInputBounds(this.yDotProduct[k], this.sigmoidLB, this.sigmoidUB);
        }

        return yActivation[0];
    }

    // Compute dot product between the inputs (including bias) and their weights
    public double dotProduct(double[] inputs, double[] weights)
    {
        double dotProd = 0.0;
        assert(inputs.length == weights.length);
        for (int i = 0; i < inputs.length; ++i) {
            dotProd += inputs[i] * weights[i];
        }
        return dotProd;
    }

    /**
    * This method does backward error propagation.
    * @param yOutput The activation of the output
    * @param yOutputCorrect The correct output for the given inputs
    * @param yDotProd The dot product at the output neurons (from forward propagation)
    * @param zActiv The activation function output of the neurons at the hidden layer (from forward propagation)
    * @param zDotProduct The dot product at the neurons at the hidden layer (from forward propagation)
    * @param xor_input The XOR inputs without the bias neuron
    * @return The updated weights from input to hidden and from hidden to output
    */
    private TwoArrays back_propagation(double[] yOutput, double[] yOutputCorrect, double[] yDotProd, double[] zActiv, double[] zDotProduct, double[] xor_input){

        double[] xorInputWithBias = concatenate(xor_input, this.bias_arr);

        double[] zOutputWithBias = concatenate(zActiv, this.bias_arr);

        // Compute delta for each output neuron
        double[] yOutputDelta = new double[this.numOutput];
        for (int k = 0; k < this.numOutput; k++){
            yOutputDelta[k] = (yOutputCorrect[k] - yOutput[k]) * customSigmoidDerivative(yDotProd[k], this.sigmoidLB, this.sigmoidUB);
        }

        // Compute weight correction terms for the weights from hidden to output
        double[][] w_hiddenToOutput_corrTerm = new double[this.numHiddenLayerNeurons+1][this.numOutput];
        for (int k = 0; k < this.numOutput; k++){
            for (int j = 0; j <= this.numHiddenLayerNeurons; j++){
                w_hiddenToOutput_corrTerm[j][k] = this.learningRate * yOutputDelta[k] * zOutputWithBias[j] + this.momentumValue * this.w_hiddenToOutput_corrTermPrev[j][k];
                // Update weights correction term at this iteration to be used as prevCorrTerm in the next iteration
                this.w_hiddenToOutput_corrTermPrev[j][k] = w_hiddenToOutput_corrTerm[j][k];
            }
        }

        // Update weights from hidden to output
        for (int k = 0; k < this.numOutput; k++){
            for (int j = 0; j <= this.numHiddenLayerNeurons; j++){
                this.w_hiddenToOutput[j][k] = this.w_hiddenToOutput[j][k] + w_hiddenToOutput_corrTerm[j][k];
            }
        }

        // Compute dot product of delta inputs (from the output layer) for each hidden layer neuron
        double[] deltaInputHidden_dotProd = new double[this.numHiddenLayerNeurons];
        for (int j=0; j < this.numHiddenLayerNeurons; j++){
            deltaInputHidden_dotProd[j] = dotProduct(yOutputDelta,  this.w_hiddenToOutput[j]);
        }

        // Compute delta error for each hidden layer neuron
        double[] deltaErrorHidden = new double[this.numHiddenLayerNeurons];
        for (int j=0; j < this.numHiddenLayerNeurons; j++) {
            deltaErrorHidden[j] = deltaInputHidden_dotProd[j] * customSigmoidDerivative(zDotProduct[j], this.sigmoidLB, this.sigmoidUB);
        }

        // Compute weight correction terms for the weights from input layer to hidden layer
        double[][] v_inputToHidden_corrTerm = new double[this.numInputs+1][this.numHiddenLayerNeurons];
        for (int j = 0; j < this.numHiddenLayerNeurons; j++){
            for (int i = 0; i <= this.numInputs; i++){
                v_inputToHidden_corrTerm[i][j] = this.learningRate*deltaErrorHidden[j] * xorInputWithBias[i] + this.momentumValue * this.v_inputToHidden_corrTermPrev[i][j];
                // Update weights correction term at this iteration to be used as prevCorrTerm in the next iteration
                this.v_inputToHidden_corrTermPrev[i][j] = v_inputToHidden_corrTerm[i][j];
            }
        }

        // Update weights from input layer to hidden layer
        for (int j = 0; j < this.numHiddenLayerNeurons; j++){
            for (int i = 0; i <= this.numInputs; i++){
                this.v_inputToHidden[i][j] = this.v_inputToHidden[i][j] + v_inputToHidden_corrTerm[i][j];
            }
        }
        return new TwoArrays(this.v_inputToHidden, this.w_hiddenToOutput);
    }

    /**
     * This method will tell the NN or the LUT the output
     * value that should be mapped to the given input vector. I.e.
     * the desired correct output value for an input.
     * @param x The input vector
     * @param argValue The new value to learn
     * @return The error in the output for that input vector
     */
    public double train(double [] x, double argValue){

        // Output of the forward propagation
        double yOutput = outputFor(x);

        // Do backward propagation
        // back_propagation() receives an array for the correct output because it was implemented with variable number of outputs
        double[] argValueVec = new double[1];
        argValueVec[0] = argValue;
        double[] yOutputVec = new double[1];
        yOutputVec[0] = yOutput;
        TwoArrays bck_ret = back_propagation(yOutputVec, argValueVec, this.yDotProduct, this.zActivation, this.zDotProduct, x);

        // Squared error
        return (argValue - yOutput) * (argValue - yOutput);

    }

    /**
    * This method implements a general sigmoid with asymptotes bounded by (a,b)
    * @param x The input
    * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
    */
    public double customSigmoidWithInputBounds(double x, double a, double b) {
        double gamma = b - a;
        double eta = -a;
        return gamma * (1.0 / (1.0 + (double) Math.exp(-x))) - eta;
    }

    public double customSigmoidDerivative(double x, double a, double b){
        double gamma = b - a;
        double eta = -a;
        // Compute custom sigmoid
        double customSigmoidValue =  customSigmoidWithInputBounds(x, a, b);
        // Compute derivative of custom sigmoid
        return 1/gamma * (eta + customSigmoidValue) * (gamma - eta - customSigmoidValue);
    }

    /**
     * Return a bipolar sigmoid of the input X
     * @param x The input
     * @return f(x) = 2 / (1+e(-x)) - 1
     */
    public double sigmoid(double x){
        return 2.0 / (1.0 + (double) Math.exp(-x)) - 1.0;
    }

    /**
     * This method implements a general sigmoid with asymptotes bounded by (a,b)
     * @param x The input
     * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
     */
    public double customSigmoid(double x){
        double a = -1.0; // lower bound
        double b = 1.0; // upper bound
        return (b - a) * (1.0 / (1.0 + (double) Math.exp(-x))) - (-a);
    }

    public double[][] initializeWeightsBetweenTwoLayers(int numInputs, int numHiddenLayer) {
        // Initialize weights as random variables in [-0.5, 0.5], including weight for the bias
        double minWeight = -0.5f;
        double maxWeight = 0.5f;
        Random rand = new Random();

        double[][] v_ij = new double[numInputs+1][numHiddenLayer];
        for (int j = 0; j < numHiddenLayer; j++){
            for (int i = 0; i <= numInputs; i++){
                v_ij[i][j] = minWeight + rand.nextFloat() * (maxWeight - minWeight);
            }
        }

        return v_ij;
    }

    /**
     * Initialize the weights to random values.
     * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
     * Like wise for hidden units. For say 2 hidden units which are stored in an array.
     * [0] & [1] are the hidden & [2] the bias.
     * We also initialise the last weight change arrays. This is to implement the alpha term.
     */
    public void initializeWeights(){
        double minWeight = -0.5f;
        double maxWeight = 0.5f;
        Random rand = new Random();

        // Initialize weights between input layer and hidden layer
        this.v_inputToHidden = new double[this.numInputs + 1][this.numHiddenLayerNeurons];
        for (int j = 0; j < this.numHiddenLayerNeurons; j++){
            for (int i = 0; i <= this.numInputs; i++){
                this.v_inputToHidden[i][j] = minWeight + rand.nextFloat() * (maxWeight - minWeight);
            }
        }

        // Initialize weights between hidden layer and output layer
        this.w_hiddenToOutput = new double[this.numHiddenLayerNeurons + 1][this.numOutput];
        for (int j = 0; j < this.numOutput; j++){
            for (int i = 0; i <= this.numHiddenLayerNeurons; i++){
                this.w_hiddenToOutput[i][j] = minWeight + rand.nextFloat() * (maxWeight - minWeight);
            }
        }

        // Initialize the weight correction terms (changes) to zero (no correction term for the first iteration)
        this.v_inputToHidden_corrTermPrev = new double[this.numInputs + 1][this.numHiddenLayerNeurons];
        this.w_hiddenToOutput_corrTermPrev = new double[this.numHiddenLayerNeurons + 1][this.numOutput];
    }

    /**
     * Initialize the weights to 0.
     */
    public void zeroWeights(){

        // Initialize weights between input layer and hidden layer
        double[][] v_ij = new double[this.numInputs+1][this.numHiddenLayerNeurons];
        for (int j = 0; j < this.numHiddenLayerNeurons; j++){
            for (int i = 0; i <= this.numInputs; i++){
                v_ij[i][j] = 0.0;
            }
        }

        // Initialize weights between hidden layer and output layer
        double[][] w_ij = new double[this.numHiddenLayerNeurons+1][this.numOutput];
        for (int j = 0; j < this.numOutput; j++){
            for (int i = 0; i <= this.numHiddenLayerNeurons; i++){
                w_ij[i][j] = 0.0;
            }
        }
    }

    private String prepare_2d_arr_for_matlab_import(double[][] arr){

        String output = "[";
        int index = 0;
        for (double[] x : arr)
        {
            output += Arrays.toString(x)+";";

            if(index+1 < arr.length ){
                output += System.lineSeparator();
            }

            index++;
        }
        output += "]";
        return output;

    }

    private void saveMSEArray(String file_path, String content) throws Exception
    {
        PrintStream out = new PrintStream(new FileOutputStream(file_path));
        out.print(content);
    }

    private double[] get_col_from_2d_arr(double[][] arr, int col_loc){

        List<Double> dbl_list = new ArrayList<Double>();
        for (double[] doubles : arr) {
            for (int col_index = 0; col_index < arr[0].length; col_index++) {
                if (col_index == col_loc) {
                    dbl_list.add(doubles[col_index]);
                }
            }
        }
        return dbl_list.stream().mapToDouble(i->i).toArray();

    }

    public double[] concatenate(double[] array1, double[] array2) {
        double[] array1and2 = new double[array1.length + array2.length];
        System.arraycopy(array1, 0, array1and2, 0, array1.length);
        System.arraycopy(array2, 0, array1and2, array1.length, array2.length);
        return array1and2;
    }

    public void print2d(double[][] arr){
        System.out.println("Printing 2D arr");
        for (double[] x : arr)
        {
            for (double y : x)
            {
                System.out.print(y + " ");
            }
            System.out.println();
        }
    }

    public static class TwoArrays {
        public final double[][] vWeights;
        public final double[][] wWeights;
        public TwoArrays(double[][] A, double[][] B) {
            this.vWeights = A;
            this.wWeights = B;
        }
    }

    /**
     * A method to write weights of an neural net to a file. Format of file contents:
     * numInputs
     * numHidden
     * weight input 0 to hidden 0
     * weight input 0 to hidden 1
     * ...
     * weight input i to hidden j
     * weight hidden 0 to output
     * weight hidden j to output
     * @param argFile of type File.
     */
    // Source: Dr. Sarkaria's code from tutorial class
    public void save(File argFile){
        PrintStream saveFile = null;

        try{
            saveFile = new PrintStream(new RobocodeFileOutputStream(argFile));
        }
        catch(IOException e){
            System.out.println("--- Coult not create output stream for NN save file.");
        }

        saveFile.println(numInputs);
        saveFile.println(numHiddenLayerNeurons);

        // Save the weights from the input to hidden neurons (one line per weight)
        // Saves the bias weight also (already in the weights array)
        for (int j = 0; j < numHiddenLayerNeurons; j++){
            for (int i = 0; i <= numInputs; i++){
                saveFile.println(v_inputToHidden[i][j]);
            }
        }

        // Save the weights from the hidden layer to the output neuron
        // Saves the bias weight also
        for (int j = 0; j < numOutput; j++){
            for (int i = 0; i <= numHiddenLayerNeurons; i++){
                saveFile.println(w_hiddenToOutput[i][j]);
            }
        }

        // Close file
        saveFile.close();

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
    public void load(String argFileName) throws IOException{

        FileInputStream inputFile = new FileInputStream(argFileName);
        BufferedReader inputReader = new BufferedReader(new InputStreamReader(inputFile));

        // Check that NN defined for file matches that created
        int numInputInFile = Integer.valueOf(inputReader.readLine());
        int numHiddenInFile = Integer.valueOf(inputReader.readLine());
        if(numInputInFile != numInputs + 1){
            System.out.println("--- Number of inputs in file is " + numInputInFile + "Expected " + numInputs + 1);
            inputReader.close();
            throw new IOException();
        }
        if(numHiddenInFile != numHiddenLayerNeurons + 1){
            System.out.println("--- Number of hidden in file is " + numHiddenInFile + "Expected " + numHiddenLayerNeurons + 1);
            inputReader.close();
            throw new IOException();
        }
        if((numInputInFile != numInputs + 1) || (numHiddenInFile != numHiddenLayerNeurons + 1)){
            inputReader.close();
            return;
        }

        // Load the weights from input layer to hidden neurons (one line per weight)
        // Loads the weights for the bias as well
        for (int j = 0; j < numHiddenLayerNeurons; j++){
            for (int i = 0; i <= numInputs; i++){
                v_inputToHidden[i][j] = Double.valueOf(inputReader.readLine());
            }
        }

        // Load the weights from the hidden layer to the output
        // Loads the weight for the bias as well
        for (int j = 0; j < numOutput; j++){
            for (int i = 0; i <= numHiddenLayerNeurons; i++){
                w_hiddenToOutput[i][j] = Double.valueOf(inputReader.readLine());
            }
        }

        // Close file
        inputFile.close();
        inputReader.close();

    }

    public void loadProcessedLuT() throws IOException {
        ProcessLuT pl = new ProcessLuT();
        pl.load_file();

        xorPatterns = pl.stateActionPatterns;
        xorExpectedOutput = pl.stateActionPatternsExpectedOutput;
        numPairs = xorExpectedOutput.length;

    }

}
