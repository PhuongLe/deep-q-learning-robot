package com.robocode;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class NeuralNetTester {
    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBinarySigmoid(){
        NeuralNet testNN = new NeuralNet(
                2,
                4,
                0.2,             // rho
                0,            // alpha
                0,                      // lower bound of sigmoid on output neuron
                1,                      // upper bound of sigmoid on output neuron
                false);

        // Start with a zero set of weights
        testNN.initializeWithZeroWeights();

        //Now apply a single training set
        double[] inputVectors = {0.0, 0.0};
        double target = 0.0;

        testNN.train(inputVectors, target);

        double [] actualLastWeightChangeHiddenToOutput = testNN.getLastWeightChangeHiddenToOutput();
        double [][] actualLastWeightChangeInputToHidden = testNN.getLastWeightChangeInputToHidden();

        double [] expectedLastWeightChangeHiddenToOutput = {0.00652, 0.00652, 0.0125};
        double [][] expectedLastWeightChangeInputToHidden = {
                {0.0, 0.0, 0.000001953125},
                {0.0, 0.0, 0.000001953125}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightChangeHiddenToOutput, actualLastWeightChangeHiddenToOutput);
        Assertions.assertArrayEquals(expectedLastWeightChangeInputToHidden, actualLastWeightChangeInputToHidden);

    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBinarySigmoid(){
        NeuralNet testNN = createNeuralNet();

        // Start with a zero set of weights
        testNN.initializeWithZeroWeights();

        //Now apply a single training set
        double[] inputVectors = {0.0, 0.0};
        double target = 0.0;

        double computedError = testNN.train(inputVectors, target);

        double expectedError = 0.241133;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBinarySigmoidWithWeights(){
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
