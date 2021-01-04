package common;

import java.io.IOException;

/**
 4 * @date 20 June 2012
 5 * @author sarbjit
 6 *
 7 */
public interface NeuralNetInterface extends CommonInterface {
    int MAX_EPOCH = 20000;
    int DID_NOT_CONVERGE = -1;

    void cloneWeights(NeuralNetInterface targetNetwork);

    double performBackPropagationTraining(double[] inputsVector, double[] expectedOutputsVector);

    /**
     * perform forward propagation for an inputs vector
     * @param inputsVector The inputs vector. An array of doubles as inputs of the neural network
     * @return the actual outputs of the corresponding inputs
     */
    double[] performFeedforward(double[] inputsVector);

    /**
     * perform backward propagation for the given errors vector
     * @param inputsVector The inputs vector. An array of doubles as inputs of the neural network
     * @param error The error corresponding to the inputs vector and the corresponding output at outputIndex
     * @param outputIndex the index of the output corresponding to the error
     */
    void performSingleErrorPropagation(double[] inputsVector, int outputIndex, double error);

    /**
     * This method implements a general sigmoid with asymptotes bounded by (a,b)
     * @param x The input
     * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
     */
    double customSigmoid(double x);

    void initializeTrainingSet() throws IOException;

    /**
     * Initialize the weights to random values.
     * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
     * Like wise for hidden units. For say 2 hidden units which are stored in an array.
     * [0] & [1] are the hidden & [2] the bias.
     * We also initialise the last weight change arrays. This is to implement the alpha term.
     */
    void initializeWeights();

    /**
     * Initialize the weights to predefined values.
     * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
     * Like wise for hidden units. For say 2 hidden units which are stored in an array.
     * [0] & [1] are the hidden & [2] the bias.
     * We also initialise the last weight change arrays. This is to implement the alpha term.
     */
    void initializeWeights(double[][] argHiddenWeight, double[][] argOutputWeight);
    /**
     * Initialize the weights and delta weights with predefined values.
     * We also initialise the last weight change arrays. This is to implement the alpha term.
     */
    void initializeWeights(double[][] argHiddenWeight, double[][] argOutputWeight, double[][] argDeltaHiddenWeight, double[][] argDeltaOutputWeight);

     /**
     * Initialize the weights to 0.
     */
     void initializeWithZeroWeights();

    /**
     * return number of epoch need for convergence
     * @param outputFileName
     * @return
     */
    int run(String outputFileName, double target, boolean showErrorAtEachEpoch, boolean showHiddenWeightsAtEachEpoch, boolean showErrorAtConverge) throws IOException;

    String printAllWeights();

    double[][] getLastWeightChangeToOutput();

    double[][] getLastWeightChangeToHidden();

    double[][] getOutputWeights();

    double[][] getHiddenWeight();

    void setActivation(Activation argActivation);
} // End of public interface NeuralNetInterface