package backpropagation;

import common.AppConfiguration;
import common.NeuralNetInterface;

import java.io.File;
import java.io.IOException;

public class StateActionSingleOutputNetworkRunner {
    private static final int NUM_TRIALS = 10;

    private static NeuralNetInterface bestNetwork = null;
    private static int bestEpochs = Integer.MAX_VALUE;

    private int train(boolean showErrorAtEachEpoch, boolean showHiddenWeightsAtEachEpoch, boolean showErrorAtConverge) throws IOException {
        double target = 0.000000005;

        NeuralNetInterface nn = new StateActionSingleOutputNetwork(true);

        nn.initializeWeights();

        int num_epoch = nn.run(AppConfiguration.RunnerReportFileName, target, showErrorAtEachEpoch, showHiddenWeightsAtEachEpoch, showErrorAtConverge);
        if (num_epoch<bestEpochs){
            bestEpochs = num_epoch;
            bestNetwork = nn;
        }

        return num_epoch;
    }

    public static void main(String []args) throws IOException {
        String showErrors = "n";
        String showHiddenWeights = "n";
        String showErrorAtConverge = "n";

        int numCoverages = 0;
        int sum = 0;
        int epochs;
        for (int i=0; i<NUM_TRIALS; i ++ ){
            StateActionSingleOutputNetworkRunner myTester = new StateActionSingleOutputNetworkRunner();
            epochs = myTester.train(showErrors.equals("y"), showHiddenWeights.equals("y"), showErrorAtConverge.equals("y"));
            if (epochs != NeuralNetInterface.DID_NOT_CONVERGE){
                numCoverages++;
                sum += epochs;
            }
        }

        if (numCoverages != 0) {
            System.out.println("-- Average convergence rate = " + sum / numCoverages);
            //System.out.println("-- Number of convergences = " + numCoverages);
            System.out.println("-- Percentage convergence rate = " + numCoverages*100/NUM_TRIALS + " %");

            bestNetwork.save(new File(AppConfiguration.NetworkWeightsFileName));
        }
        else {
            System.out.println("-- Cannot reach the target after " + NUM_TRIALS + " tries");
        }
    }
}