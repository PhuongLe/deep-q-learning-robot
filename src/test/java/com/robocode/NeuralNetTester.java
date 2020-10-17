package com.robocode;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class NeuralNetTester {
    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet000(){
        NeuralNet testNN = createBinaryNeuralNet();

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

        double [] expectedLastWeightChangeToOutput = {-0.017178231, -0.017178231, -0.017178231, -0.017178231};
        //double [] expectedLastWeightChangeToOutput = {-0.015114732, -0.015114732, -0.015114732, -0.015114732};
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
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet101(){
        NeuralNet testNN = createBinaryNeuralNet();

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

        double [] expectedLastWeightToOutput = {0.314792222, 0.564953289, 0.134655362, 0.294967576};
        double [][] expectedLastWeightToHidden = {
                {0.230470923, 0.340808872, 0.140208003, 0.350420533},
                {0.52, 0.27, 0.19, 0.25}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput, 0.00001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.00001);
        }
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBinarySigmoidWithZeroWeightsWithTrainingSet000(){
        NeuralNet testNN = createBinaryNeuralNet();

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
    public void testOneForwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet101(){
        NeuralNet testNN = createBinaryNeuralNet();

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
    public void testOneForwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet000(){
        NeuralNet testNN = createBinaryNeuralNet();

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

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM1M1M1(){
        NeuralNet testNN = createBipolarNeuralNet();

        // Start with a zero set of weights
        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputHiddenWeights = {0.31, 0.56, 0.13,0.29};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {-1.0, -1.0};
        double target = -1.0;

        double computedError = testNN.train(inputVectors, target);

        double expectedError = -1.190506994;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError, 0.0001);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM1M1M1() {
        NeuralNet testNN = createBipolarNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputHiddenWeights = {0.31, 0.56, 0.13,0.29};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {-1.0, -1.0};
        double target = -1.0;

        testNN.train(inputVectors, target);

        double [] actualLastWeightToOutput = testNN.getLastWeightToOutput();
        double [][] actualLastWeightToHidden = testNN.getLastWeightToHidden();

        double [] expectedLastWeightToOutput = {0.316955619, 0.561835547, 0.121591758, 0.291468548};
        double [][] expectedLastWeightToHidden = {
                {0.225173132, 0.337885896, 0.141810631, 0.34912681},
                {0.515173132, 0.267885896, 0.191810631, 0.24912681}
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
    public void testOneForwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM111(){
        NeuralNet testNN = createBipolarNeuralNet();

        // Start with a zero set of weights
        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputHiddenWeights = {0.31, 0.56, 0.13,0.29};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {-1.0, 1.0};
        double target = 1.0;

        double computedError = testNN.train(inputVectors, target);

        double expectedError = 0.627397447;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError, 0.0001);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM111() {
        NeuralNet testNN = createBipolarNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputHiddenWeights = {0.31, 0.56, 0.13,0.29};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {-1.0, 1.0};
        double target = 1.0;

        testNN.train(inputVectors, target);

        double [] actualLastWeightToOutput = testNN.getLastWeightToOutput();
        double [][] actualLastWeightToHidden = testNN.getLastWeightToHidden();

        double [] expectedLastWeightToOutput = {0.333533206, 0.574092485, 0.147358383, 0.303260327};
        double [][] expectedLastWeightToHidden = {
                {0.22059888, 0.327706052, 0.136397866, 0.343775208},
                {0.52940112, 0.282293948, 0.193602134, 0.256224792}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput, 0.0001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.0001);
        }
    }

    @Test
    public void testOneBackwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM1M1M11() {
        NeuralNet testNN = createBipolarNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.420346613, 0.36364129, -0.127700141, -0.239620821},
                {0.048133905, -0.261157496, 0.101547675, 0.119521338}
        };

        double[] argOutputHiddenWeights = {0.346084656, -0.261350465, 0.147112986,0.088036798};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);
        testNN.initializeBias(0.0, 0.0);

        //Now apply a single training set
        double[] inputVectors = {-1.0, -1.0};
        double target = -1.0;

        double computedError = testNN.train(inputVectors, target);

        double expectedError = -0.970492609;

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
        NeuralNet testNN = createBinaryNeuralNet();

        double computedActivation = testNN.computeActivation(0.023);
        double expectedValue = 0.505749747;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedValue, computedActivation, 0.000001);
    }

    private NeuralNet createBipolarNeuralNet(){
        return new NeuralNet(
                2,
                4,
                0.2,             // rho
                0,            // alpha
                -1,                      // lower bound of sigmoid on output neuron
                1,                      // upper bound of sigmoid on output neuron
                false);
    }

    private NeuralNet createBinaryNeuralNet(){
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
