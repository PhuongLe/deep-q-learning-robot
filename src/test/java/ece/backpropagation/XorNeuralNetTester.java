package ece.backpropagation;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;

public class XorNeuralNetTester {
    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet000() throws IOException {
        XorNeuralNet testNN = createBinaryNeuralNet();

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
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet101() throws IOException {
        XorNeuralNet testNN = createBinaryNeuralNet();

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

        double [] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

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
    public void testOneForwardPropagationWithBinarySigmoidWithZeroWeightsWithTrainingSet000() throws IOException {
        XorNeuralNet testNN = createBinaryNeuralNet();

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
    public void testOneForwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet101() throws IOException {
        XorNeuralNet testNN = createBinaryNeuralNet();

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
    public void testOneForwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet000() throws IOException {
        XorNeuralNet testNN = createBinaryNeuralNet();

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
    public void testOneForwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM1M1M1() throws IOException {
        XorNeuralNet testNN = createBipolarNeuralNet();

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
    public void testOneBackwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM1M1M1() throws IOException {
        XorNeuralNet testNN = createBipolarNeuralNet();

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

        double [] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.320866679, 0.562867653, 0.116863907, 0.292294294};
        double [][] expectedLastWeightToHidden = {
                {0.248241393, 0.372268731, 0.146616015, 0.366760757},
                {0.538241393, 0.302268731, 0.196616015, 0.266760757}
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
    public void testOneForwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM111() throws IOException {
        XorNeuralNet testNN = createBipolarNeuralNet();

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
    public void testOneBackwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM111() throws IOException {
        XorNeuralNet testNN = createBipolarNeuralNet();

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

        double [] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.331673052, 0.572978561, 0.145986311, 0.302212181};
        double [][] expectedLastWeightToHidden = {
                {0.222481699, 0.325414315, 0.136401484, 0.342252924},
                {0.527518301, 0.284585685, 0.193598516, 0.257747076}
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
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet101WithMomentum09() throws IOException {
        XorNeuralNet testNN = createBinaryNeuralNet();
        testNN.momentumTerm = 0.9;

        double[][] argDeltaHiddenWeight = {
                {0.001, 0.034, 0.0021, 0.0012},
                {0.0032, 0.0025, 0.0037, 0.00112}
        };
        double[] argDeltaOutputWeight = {0.00345, 0.00267, 0.0021,0.0045};

        double[][] argHiddenWeight = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputWeight = {0.31, 0.56, 0.13,0.29};

        testNN.initializeWeights(argHiddenWeight, argOutputWeight, argDeltaHiddenWeight, argDeltaOutputWeight);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {1.0, 0.0};
        double target = 1.0;

        testNN.train(inputVectors, target);

        double [] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.317897222, 0.567356289, 0.136545362, 0.299017576};
        double [][] expectedLastWeightToHidden = {
                {0.231375568338, 0.371412312774, 0.142100922445, 0.351506307458},
                {0.52288, 0.27225, 0.19333, 0.251008}
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
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet000WithMomentum09() throws IOException {
        XorNeuralNet testNN = createBinaryNeuralNet();
        testNN.momentumTerm = 0.9;

        double[][] argDeltaHiddenWeight = {
                {0.001, 0.034, 0.0021, 0.0012},
                {0.0032, 0.0025, 0.0037, 0.00112}
        };
        double[] argDeltaOutputWeight = {0.00345, 0.00267, 0.0021,0.0045};

        double[][] argHiddenWeight = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputWeight = {0.31, 0.56, 0.13,0.29};

        testNN.initializeWeights(argHiddenWeight, argOutputWeight, argDeltaHiddenWeight, argDeltaOutputWeight);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {0.0, 0.0};
        double target = 0.0;

        testNN.train(inputVectors, target);

        double [] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.295926769, 0.545224769, 0.114711769, 0.276871769};
        double [][] expectedLastWeightToHidden = {
                {0.230900000000, 0.370600000000, 0.141890000000, 0.351080000000},
                {0.52288, 0.27225, 0.19333, 0.251008}
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
    public void testOneBackwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSet101WithMomentum09() throws IOException {
        XorNeuralNet testNN = createBipolarNeuralNet();
        testNN.momentumTerm = 0.9;

        double[][] argDeltaHiddenWeight = {
                {0.001, 0.034, 0.0021, 0.0012},
                {0.0032, 0.0025, 0.0037, 0.00112}
        };
        double[] argDeltaOutputWeight = {0.00345, 0.00267, 0.0021,0.0045};

        double[][] argHiddenWeight = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25}
        };
        double[] argOutputWeight = {0.31, 0.56, 0.13,0.29};

        testNN.initializeWeights(argHiddenWeight, argOutputWeight, argDeltaHiddenWeight, argDeltaOutputWeight);
        testNN.initializeBias(0.56, 0.42);

        //Now apply a single training set
        double[] inputVectors = {-1.0, -1.0};
        double target = -1.0;

        testNN.train(inputVectors, target);

        double [] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.323971679, 0.565270653, 0.118753907, 0.296344294};
        double [][] expectedLastWeightToHidden = {
                {0.249317913, 0.403006493, 0.148613014, 0.368072993},
                {0.541297913, 0.304656493, 0.200053014, 0.268000993}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput, 0.00001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.00001);
        }
    }


    @Test
    public void testBipolarSigmoid(){

    }

    @Test
    public void testBinarySigmoid(){

    }

    @Test
    public void testActivationFunction() throws IOException {
        XorNeuralNet testNN = createBinaryNeuralNet();

        double computedActivation = testNN.computeActivation(0.023);
        double expectedValue = 0.505749747;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedValue, computedActivation, 0.000001);
    }

    private XorNeuralNet createBipolarNeuralNet() throws IOException {
        return new XorNeuralNet(
                2,
                4,
                0.2,             // rho
                0,            // alpha
                -1,                      // lower bound of sigmoid on output neuron
                1,                      // upper bound of sigmoid on output neuron
                true);
    }

    private XorNeuralNet createBinaryNeuralNet() throws IOException {
        return new XorNeuralNet(
                2,
                4,
                0.2,             // rho
                0,            // alpha
                0,                      // lower bound of sigmoid on output neuron
                1,                      // upper bound of sigmoid on output neuron
                false);
    }
}
