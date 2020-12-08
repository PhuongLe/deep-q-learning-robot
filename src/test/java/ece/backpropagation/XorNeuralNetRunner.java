    package ece.backpropagation;

    import ece.common.NeuralNetInterface;

    import java.io.*;
    import java.time.LocalDateTime;
    import java.time.format.DateTimeFormatter;
    import java.util.Scanner;

    public class XorNeuralNetRunner {
        static DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");

        static int argumentA;
        static int argumentB;
        static boolean argUseBipolarHiddenNeurons = false;
        static double hiddenBias = 6.0;
        static double outputBias = 6.0;

        static double argMomentum = 0.0;

        private int train(String outputFileName, boolean showErrorAtEachEpoch, boolean showHiddenWeightsAtEachEpoch, boolean showErrorAtConverge) throws IOException {
            double target = 0.05;

            XorNeuralNet nn = new XorNeuralNet(
                    2,
                    4,
                    0.2,                // rho
                    argMomentum,                    // alpha
                    argumentA,                      // lower bound of sigmoid on output neuron
                    argumentB,                      // upper bound of sigmoid on output neuron
                    argUseBipolarHiddenNeurons,
                    true);

            nn.initializeWeights();
            //nn.initializeBias(hiddenBias, outputBias);
            //nn.enableBatchUpdateOption();

            //Activation activation = new TanhActivation();
            //Activation activation = new ReLuActivation();
            //nn.setActivation(activation);

            return nn.run(outputFileName, target, showErrorAtEachEpoch, showHiddenWeightsAtEachEpoch, showErrorAtConverge);
        }

        public static void main(String []args) throws IOException {
            Scanner reader = new Scanner(System.in);
            System.out.print("Enter the option you want to run 0.binary 1.bipolar: ");
            int option = reader.nextInt();
            /*System.out.print("Enter the number of trials you want to run: ");
            int numTrials = reader.nextInt();
            System.out.print("Do you want to see error at each epoch y/n?: ");
            String showErrors = reader.next();
            System.out.print("Do you want to see the hidden weights at each epoch y/n?: ");
            String showHiddenWeights = reader.next();
            System.out.print("Do you want to see error at converge y/n?: ");
            String showErrorAtConverge = reader.next();*/

            int numTrials = 100;
            String showErrors = "n";
            String showHiddenWeights = "n";
            String showErrorAtConverge = "n";
            String fileNameSuffix = "binary";
            switch (option){
                case 0:{
                    argumentA = 0;
                    argumentB = 1;
                    argUseBipolarHiddenNeurons = false;
                    fileNameSuffix = "binary";
                    break;
                }
                case 1:{
                    argumentA = -1;
                    argumentB = 1;
                    argUseBipolarHiddenNeurons = true;
                    fileNameSuffix = "bipolar";
                }
            }

            LocalDateTime now = LocalDateTime.now();
            String fileName = "Report\\ConvergeResult_" + fileNameSuffix + dtf.format(now) + ".csv";

            int numCoverages = 0;
            int sum = 0;
            int epochs;
            for (int i=0; i<numTrials; i ++ ){
                XorNeuralNetRunner myTester = new XorNeuralNetRunner();
                epochs = myTester.train(fileName, showErrors.equals("y"), showHiddenWeights.equals("y"), showErrorAtConverge.equals("y"));
                if (epochs != NeuralNetInterface.DID_NOT_CONVERGE){
                    numCoverages++;
                    sum += epochs;
                }
            }

            if (numCoverages != 0) {
                System.out.println("-- Average convergence rate = " + sum / numCoverages);
                //System.out.println("-- Number of convergences = " + numCoverages);
                System.out.println("-- Percentage convergence rate = " + numCoverages*100/numTrials + " %");
            }
            else {
                System.out.println("-- Cannot reach the target after " + numTrials + " tries");
            }
        }
    }