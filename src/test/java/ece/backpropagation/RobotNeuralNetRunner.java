    package ece.backpropagation;

    import ece.common.NeuralNetInterface;

    import java.io.File;
    import java.io.IOException;
    import java.text.SimpleDateFormat;
    import java.util.Date;

    public class RobotNeuralNetRunner {
        static String baseFolder = "d:\\Google Drive\\LXP\\UBC\\Term 3\\CPEN 502 - ML\\Assignments\\Robocode\\out\\report\\" + new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss") .format(new Date());
        static String runnerReportFileName = baseFolder+ "-runner.log";
        static String nnWeightsFileName = baseFolder+ "-nn_weights.dat";

        private static final int NUM_TRIALS = 10;

        static double HIDDEN_BIAS = 1.0;
        static double OUTPUT_BIAS = 1.0;

        private static NeuralNetInterface bestNetwork = null;
        private static int bestEpochs = Integer.MAX_VALUE;

        private int train(boolean showErrorAtEachEpoch, boolean showHiddenWeightsAtEachEpoch, boolean showErrorAtConverge) throws IOException {
            double target = 0.00005;

            NeuralNetInterface nn = new StateActionNeuralNet();

            nn.initializeWeights();
            nn.initializeBias(HIDDEN_BIAS, OUTPUT_BIAS);

            int num_epoch = nn.run(runnerReportFileName, target, showErrorAtEachEpoch, showHiddenWeightsAtEachEpoch, showErrorAtConverge);
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
                RobotNeuralNetRunner myTester = new RobotNeuralNetRunner();
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

                bestNetwork.save(new File(nnWeightsFileName));
            }
            else {
                System.out.println("-- Cannot reach the target after " + NUM_TRIALS + " tries");
            }
        }
    }