package com.robocode;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class NeuralNetTester {
    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWith000(){
        NeuralNet testNN = createNeuralNet();

        // Start with a zero set of weights
        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputHiddenWeights = {0.31, 0.56, 0.13,0.29};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {0.0, 0.0};
        double target = 0.0;

        testNN.train(inputVectors, target);

        double [] actualLastWeightChangeToOutput = testNN.getLastWeightChangeToOutput();
        double [][] actualLastWeightChangeToHidden = testNN.getLastWeightChangeToHidden();

        double [] expectedLastWeightChangeToOutput = {0.005862567, 0.005862567, 0.005862567, 0.005862567};
        double [][] expectedLastWeightChangeToHidden = {
                {0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightChangeToOutput, actualLastWeightChangeToOutput, 0.00001);
        Assertions.assertArrayEquals(expectedLastWeightChangeToHidden, actualLastWeightChangeToHidden);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWith101(){
        NeuralNet testNN = createNeuralNet();

        // Start with a zero set of weights
        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputHiddenWeights = {0.31, 0.56, 0.13,0.29};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {1.0, 0.0};
        double target = 1.0;

        testNN.train(inputVectors, target);

        double [] actualLastWeightToOutput = testNN.getLastWeightToOutput();
        double [][] actualLastWeightToHidden = testNN.getLastWeightToHidden();

        double [] expectedLastWeightToOutput = {0.310577894, 0.56065836, 0.130512058, 0.290665675};
        double [][] expectedLastWeightToHidden = {
                {0.230048782, 0.340084281, 0.140021167, 0.35004351},
                {0.52, 0.27, 0.19, 0.25}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput, 0.0001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.0001);
        }
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBinarySigmoidWithZeroWeightsWith000(){
        NeuralNet testNN = createNeuralNet();

        // Start with a zero set of weights
        testNN.initializeWithZeroWeights();

        //Now apply a single training set
        double[] inputVectors = {0.0, 0.0};
        double target = 0.0;

        double computedError = testNN.train(inputVectors, target);

        double expectedError = 0.125;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBinarySigmoidWithWeightsWith101(){
        NeuralNet testNN = createNeuralNet();

        // Start with a zero set of weights
        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputHiddenWeights = {0.31, 0.56, 0.13,0.29};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {1.0, 0.0};
        double target = 1.0;

        double computedError = testNN.train(inputVectors, target);

        double expectedError = 0.022047614;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError, 0.0001);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBinarySigmoidWithWeightsWith000(){
        NeuralNet testNN = createNeuralNet();

        // Start with a zero set of weights
        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputHiddenWeights = {0.31, 0.56, 0.13,0.29};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {0.0, 0.0};
        double target = 0.0;

        double computedError = testNN.train(inputVectors, target);

        double expectedError = 0.300887955;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError, 0.0001);
    }

    @Test
    public void testBipolarSigmoid(){

    }

    @Test
    public void testBinarySigmoid(){

    }

    @Test
    public void testActivationFunction(){
        NeuralNet testNN = createNeuralNet();

        double computedActivation = testNN.executeActivation(0.023);
        double expectedValue = 0.23;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedValue, computedActivation);
    }

    private NeuralNet createNeuralNet(){
        return new NeuralNet(
                2,
                4,
                0.2,             // rho
                0,            // alpha
                0,                      // lower bound of sigmoid on output neuron
                1,                      // upper bound of sigmoid on output neuron
                false);
    }
}
