package backpropagation;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class XorNeuralNetTester {
    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet000() {
        XorNeuralNet testNN = createBinaryNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };

        double[][] argOutputHiddenWeights = {{0.31, 0.56, 0.13,0.29, 0.42}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);

        //Now apply a single training set
        double[] inputVectors = {0.0, 0.0, 1.0};
        double[] target = {0.0};

        testNN.train(inputVectors, target);

        double [][] actualLastWeightChangeToOutput = testNN.getLastWeightChangeToOutput();
        double [][] actualLastWeightChangeToHidden = testNN.getLastWeightChangeToHidden();

        double [] expectedLastWeightChangeToOutput = {-0.017178231, -0.017178231, -0.017178231, -0.017178231, -0.026990592};
        double [][] expectedLastWeightChangeToHidden = {
                {0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0},
                {-0.001828702, -0.003389977, -0.000704583, -0.0017038}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightChangeToOutput, actualLastWeightChangeToOutput[0], 0.00001);
        for (int i=0;  i < expectedLastWeightChangeToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightChangeToHidden[i], actualLastWeightChangeToHidden[i], 0.00001);
        }
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet101(){
        XorNeuralNet testNN = createBinaryNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };

        double[][] argOutputHiddenWeights = {{0.31, 0.56, 0.13,0.29, 0.42}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);

        //Now apply a single training set
        double[] inputVectors = {1.0, 0.0, 1.0};
        double[] target = {1.0};

        testNN.train(inputVectors, target);

        double [][] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.314792222, 0.564953289, 0.134655362, 0.294967576, 0.426967146};
        double [][] expectedLastWeightToHidden = {
                {0.230470923, 0.340808872, 0.140208003, 0.350420533},
                {0.52, 0.27, 0.19, 0.25},
                {0.560470923, 0.560808872, 0.560208003, 0.560420533}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput[0], 0.00001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.00001);
        }
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBinarySigmoidWithZeroWeightsWithTrainingSet000(){
        XorNeuralNet testNN = createBinaryNeuralNet();

        // Start with a zero set of weights
        testNN.initializeWithZeroWeights();

        //Now apply a single training set
        double[] inputVectors = {0.0, 0.0, 1.0};
        double[] target = {0.0};

        double computedError = testNN.train(inputVectors, target);

        //double expectedError = -0.731058579;
        double expectedError = 0.125;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError, 0.0001);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet101(){
        XorNeuralNet testNN = createBinaryNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };

        double[][] argOutputHiddenWeights = {{0.31, 0.56, 0.13,0.29, 0.42}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);

        //Now apply a single training set
        double[] inputVectors = {1.0, 0.0, 1.0};
        double[] target = {1.0};

        double computedError = testNN.train(inputVectors, target);

        //double expectedError = 0.20998864;
        double expectedError = 0.022047614;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError, 0.0001);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet000(){
        XorNeuralNet testNN = createBinaryNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };

        double[][] argOutputWeights = {{0.31, 0.56, 0.13,0.29, 0.42}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputWeights);

        //Now apply a single training set
        double[] inputVectors = {0.0, 0.0, 1.0};
        double[] target = {0.0};

        double computedError = testNN.train(inputVectors, target);

        double expectedError = 0.300887955;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError, 0.0001);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM1M1M1(){
        XorNeuralNet testNN = createBipolarNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };
        double[][] argOutputHiddenWeights = {{0.31, 0.56, 0.13,0.29, 0.42}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);

        //Now apply a single training set
        double[] inputVectors = {-1.0, -1.0, 1.0};
        double[] target = {-1.0};

        double computedError = testNN.train(inputVectors, target);

        //double expectedError = -1.190506994;
        double expectedError = 0.708653451;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError, 0.0001);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM1M1M1() {
        XorNeuralNet testNN = createBipolarNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };

        double[][] argOutputHiddenWeights = {{0.31, 0.56, 0.13,0.29, 0.42}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);

        //Now apply a single training set
        double[] inputVectors = {-1.0, -1.0, 1.0};
        double[] target = {-1.0};

        testNN.train(inputVectors, target);

        double [][] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.320866679, 0.562867653, 0.116863907, 0.292294294, 0.305269997};
        double [][] expectedLastWeightToHidden = {
                {0.248241393, 0.372268731, 0.146616015, 0.366760757},
                {0.538241393, 0.302268731, 0.196616015, 0.266760757},
                {0.541758607, 0.527731269, 0.553383985, 0.543239243}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput[0], 0.0001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.0001);
        }
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneForwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM111(){
        XorNeuralNet testNN = createBipolarNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };

        double[][] argOutputHiddenWeights = {{0.31, 0.56, 0.13,0.29, 0.42}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);

        //Now apply a single training set
        double[] inputVectors = {-1.0, 1.0, 1.0};
        double[] target = {1.0};

        double computedError = testNN.train(inputVectors, target);

        //double expectedError = 0.627397447;
        double expectedError = 0.196813778;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedError, computedError, 0.0001);
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSetM111() {
        XorNeuralNet testNN = createBipolarNeuralNet();

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };

        double[][] argOutputHiddenWeights = {{0.31, 0.56, 0.13,0.29, 0.42}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);

        //Now apply a single training set
        double[] inputVectors = {-1.0, 1.0, 1.0};
        double[] target = {1.0};

        testNN.train(inputVectors, target);

        double [][] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.331673052, 0.572978561, 0.145986311, 0.302212181, 0.474029419};
        double [][] expectedLastWeightToHidden = {
                {0.222481699, 0.325414315, 0.136401484, 0.342252924},
                {0.527518301, 0.284585685, 0.193598516, 0.257747076},
                {0.567518301, 0.574585685, 0.563598516, 0.567747076}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput[0], 0.0001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.0001);
        }
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet101WithMomentum09() {
        XorNeuralNet testNN = createBinaryNeuralNet();
        testNN.momentumTerm = 0.9;

        double[][] argDeltaHiddenWeight = {
                {0.001, 0.034, 0.0021, 0.0012},
                {0.0032, 0.0025, 0.0037, 0.00112},
                {0.0,0.0,0.0,0.0}
        };
        double[][] argDeltaOutputWeight = {{0.00345, 0.00267, 0.0021,0.0045, 0.0001}};

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };

        double[][] argOutputHiddenWeights = {{0.31, 0.56, 0.13,0.29, 0.42}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);

        //Now apply a single training set
        double[] inputVectors = {1.0, 0.0, 1.0};
        double[] target = {1.0};

        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights, argDeltaHiddenWeight, argDeltaOutputWeight);

        testNN.train(inputVectors, target);

        double [][] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.317897222, 0.567356289, 0.136545362, 0.299017576, 0.427057146};
        double [][] expectedLastWeightToHidden = {
                {0.231375568338, 0.371412312774, 0.142100922445, 0.351506307458},
                {0.52288, 0.27225, 0.19333, 0.251008},
                {0.560475568, 0.560812313, 0.560210922, 0.560426307}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput[0], 0.00001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.00001);
        }
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBinarySigmoidWithWeightsWithTrainingSet000WithMomentum09() {
        XorNeuralNet testNN = createBinaryNeuralNet();
        testNN.momentumTerm = 0.9;

        double[][] argDeltaHiddenWeight = {
                {0.001, 0.034, 0.0021, 0.0012},
                {0.0032, 0.0025, 0.0037, 0.00112},
                {0.0,0.0,0.0,0.0}
        };
        double[][] argDeltaOutputWeight = {{0.00345, 0.00267, 0.0021,0.0045, 0.0001}};

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };

        double[][] argOutputHiddenWeights = {{0.31, 0.56, 0.13, 0.29, 0.42}};

        //Now apply a single training set
        double[] inputVectors = {0.0, 0.0, 1.0};
        double[] target = {0.0};

        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights, argDeltaHiddenWeight, argDeltaOutputWeight);

        testNN.train(inputVectors, target);

        double [][] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.295926769, 0.545224769, 0.114711769, 0.276871769, 0.393099408};
        double [][] expectedLastWeightToHidden = {
                {0.2309, 0.3706, 0.14189, 0.35108},
                {0.52288, 0.27225, 0.19333, 0.251008},
                {0.558151907, 0.556595016, 0.559283613, 0.558270907}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput[0], 0.00001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.00001);
        }
    }

    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testOneBackwardPropagationWithBipolarSigmoidWithWeightsWithTrainingSet101WithMomentum09() {
        XorNeuralNet testNN = createBipolarNeuralNet();
        testNN.momentumTerm = 0.9;

        double[][] argDeltaHiddenWeight = {
                {0.001, 0.034, 0.0021, 0.0012},
                {0.0032, 0.0025, 0.0037, 0.00112},
                {0.0,0.0,0.0,0.0}
        };
        double[][] argDeltaOutputWeight = {{0.00345, 0.00267, 0.0021,0.0045, 0.0001}};

        double[][] argInputHiddenWeights = {
                {0.23, 0.34, 0.14, 0.35},
                {0.52, 0.27, 0.19, 0.25},
                {0.56, 0.56, 0.56, 0.56}
        };

        double[][] argOutputHiddenWeights = {{0.31, 0.56, 0.13, 0.29, 0.42}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);

        //Now apply a single training set
        double[] inputVectors = {-1.0, -1.0, 1.0};
        double[] target = {-1.0};

        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights, argDeltaHiddenWeight, argDeltaOutputWeight);

        testNN.train(inputVectors, target);

        double [][] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.323971679, 0.565270653, 0.118753907, 0.296344294, 0.305359997};
        double [][] expectedLastWeightToHidden = {
                {0.249317913, 0.403006493, 0.148613014, 0.368072993},
                {0.541297913, 0.304656493, 0.200053014, 0.268000993},
                {0.541582087, 0.527593507, 0.553276986, 0.543007007}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput[0], 0.001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.00001);
        }
    }

    @Test
    public void testActivationFunction() {
        XorNeuralNet testNN = createBinaryNeuralNet();

        double computedActivation = testNN.computeActivation(0.023);
        double expectedValue = 0.505749747;

        //Test computed error on output neuron
        Assertions.assertEquals(expectedValue, computedActivation, 0.000001);
    }


    /**
     * This is a very basic test, which looks at the very first weight updates from initial weights
     */
    @Test
    public void testRandomScheme() {
        XorNeuralNet testNN = createBipolarNeuralNet();
        testNN.learningRate = 0.02;
        testNN.momentumTerm = 0.0;

        double[][] argDeltaHiddenWeight = {
                {0.002253179, 0.001277793, 0.002018392, -0.001581669},
                {0.002253179, 0.001277793, 0.002018392, -0.001581669},
                {-0.002253179, -0.001277793, -0.002018392, 0.001581669}
        };
        double[][] argDeltaOutputWeight = {{0.002103128, -5.93E-04, 0.002796272, 2.24E-05, -0.010574471}};

        double[][] argInputHiddenWeights = {
                {-0.239455017, 0.203991047, 0.202025099, -0.194889069},
                {0.285918666, 0.080075874, -0.090382033, 0.038235083},
                {-0.363443355, 0.392495922, -0.436156346, -0.156137389}
        };

        double[][] argOutputHiddenWeights = {{0.443705752, 0.242437392, 0.410449307, -0.299149886, 0.293860214}};
        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights);

        //Now apply a single training set
        double[] inputVectors = {-1.0, 1.0, 1.0};
        double[] target = {1.0};

        testNN.initializeWeights(argInputHiddenWeights, argOutputHiddenWeights, argDeltaHiddenWeight, argDeltaOutputWeight);

        testNN.train(inputVectors, target);

        double [][] actualLastWeightToOutput = testNN.getOutputWeights();
        double [][] actualLastWeightToHidden = testNN.getHiddenWeight();

        double [] expectedLastWeightToOutput = {0.444422463, 0.243621628, 0.407353359, -0.298808564, 0.302731635};
        double [][] expectedLastWeightToHidden = {
                {-0.24141348, 0.202929668, 0.200438255, -0.193565602},
                {0.287877129, 0.081137253, -0.088795188, 0.036911617},
                {-0.361484892, 0.393557301, -0.434569501, -0.157460856}
        };

        //Test weights on output neuron
        Assertions.assertArrayEquals(expectedLastWeightToOutput, actualLastWeightToOutput[0], 0.001);
        for (int i=0;  i < expectedLastWeightToHidden.length; i++) {
            Assertions.assertArrayEquals(expectedLastWeightToHidden[i], actualLastWeightToHidden[i], 0.00001);
        }
    }

    private XorNeuralNet createBipolarNeuralNet() {
        return new XorNeuralNet(
                2,
                4,
                1,
                0.2,             // rho
                0,            // alpha
                -1,                      // lower bound of sigmoid on output neuron
                1,                      // upper bound of sigmoid on output neuron
                true,
                false);
    }

    private XorNeuralNet createBinaryNeuralNet() {
        return new XorNeuralNet(
                2,
                4,
                1,
                0.2,             // rho
                0,            // alpha
                0,                      // lower bound of sigmoid on output neuron
                1,                      // upper bound of sigmoid on output neuron
                false,
                false);
    }
}
