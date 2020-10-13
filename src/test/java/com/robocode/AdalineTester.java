package com.robocode;

import com.robocode.Adaline;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;


public class AdalineTester {
    @Test
    public void testOutput(){
        double [] inputVector = new double[]{0.5, 0.6};
        double [] testWeights = new double[]{1.2, 0.4, -0.96};
        Adaline testAdeline = new Adaline(2);
        testAdeline.setWeights(testWeights);

        double expectedOutput = 1.1;
        double actualOutput = testAdeline.Output(inputVector);

        Assertions.assertEquals(expectedOutput, actualOutput, 0.001);
    }

    @Test
    public void testLoss(){
        double [][] trainingInputVectors = {
            {0.0, 0.0},
            {0.0, 0.1},
            {0.1,0.0},
            {1.0,1.1}
        };
        double [] trainingTargetVectors = {0.0, 0.0, 0.0, 0.1};

        double [] testWeights = new double[]{1.2, 0.4, -0.96};
        Adaline testAdeline = new Adaline(2);
        testAdeline.setWeights(testWeights);

        double expectedLoss = 4.02;
        double actualLoss = testAdeline.loss(trainingInputVectors, trainingTargetVectors);
        Assertions.assertEquals(expectedLoss, actualLoss, 0.001);
    }
}
