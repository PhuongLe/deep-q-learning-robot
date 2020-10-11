package com.cpen502.robocode;

import java.util.Scanner;

public class NeuralNetRunner {
    private static final int DID_NOT_CONVERGE = -1;

    double [][] xorTrainingSet = {{0,0}, {0,1}, {1,0}, {1,1}};
    double [] xorTargetSet = {0, 1, 1, 0};

    NeuralNet nn = new NeuralNet(
            2,
            4,
            0.2,             // rho
            0,            // alpha
            0,                      // lower bound of sigmoid on output neuron
            1,                      // upper bound of sigmoid on output neuron
            false);// use bipolar sigmoid for hidden neurons?

    private double totalError(){
        double sumError = 0;
        double output = 0;
        for (int i = 0; i< xorTargetSet.length; i++){
            output = nn.outputFor(xorTrainingSet[i]);
            sumError += 0.5 * Math.pow(xorTargetSet[i] - output, 2);
        }
        return sumError;
    }

    private int train(boolean showErrorAtEachEpoch, boolean showHiddenWeightsAtEachEpoch){
        return 0;
    }

    public static void main(String []args){
        Scanner reader = new Scanner(System.in);
        System.out.print("Enter the number of trials you want to run: ");
        int numTrials = reader.nextInt();
        System.out.print("Do you want to see error at each epoch y/n?: ");
        String showErrors = reader.next();
        System.out.print("Do you want to see the hidden weights at each epoch y/n?: ");
        String showHiddenWeights = reader.next();

        int numCoveraged = 0;
        int sum = 0;
        int epochs = 0;
        for (int i=0; i<numTrials; i ++ ){
            NeuralNetRunner myTester = new NeuralNetRunner();
            epochs = myTester.train(showErrors.equals('y'), showHiddenWeights.equals('y'));
            if (epochs != DID_NOT_CONVERGE){
                numCoveraged++;
                sum += epochs;
            }
        }

        System.out.println("-- Average convergence rate = "+ (int) sum/numCoveraged);
    }
}