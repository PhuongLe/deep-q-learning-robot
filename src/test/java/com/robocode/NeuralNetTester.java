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

        double [] expectedLastWeightChangeToOutput = {-0.015114732, -0.015114732, -0.015114732, -0.015114732};
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

        double [] expectedLastWeightToOutput = {0.315504046, 0.566270432, 0.134877002, 0.296340103};
        double [][] expectedLastWeightToHidden = {
                {0.230471988, 0.340810758, 0.140208345, 0.35042249},
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

        double expectedError = -0.5;

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

        double expectedError = 0.20998864;

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

        double expectedError = -0.775742167;

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
