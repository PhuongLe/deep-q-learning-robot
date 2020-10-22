    package ece.backpropagation;

    import java.io.*;
    import java.time.LocalDateTime;
    import java.time.format.DateTimeFormatter;
    import java.util.ArrayList;
    import java.util.List;
    import java.util.Scanner;

    public class NeuralNetRunner {
        private static final int DID_NOT_CONVERGE = -1;
        static int MAX_EPOCH = 20000;
        static DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");

        static int argumentA;
        static int argumentB;
        static boolean argUseBipolarHiddenNeurons = false;
        static double hiddenBias = 6.0;
        static double outputBias = 6.0;

        static double argMomentum = 0.0;

        private int train(PrintWriter output, boolean showErrorAtEachEpoch, boolean showHiddenWeightsAtEachEpoch, boolean showErrorAtConverge) {
            double target = 0.05;

            NeuralNet nn = new NeuralNet(
                    2,
                    4,
                    0.2,                // rho
                    argMomentum,                    // alpha
                    argumentA,                      // lower bound of sigmoid on output neuron
                    argumentB,                      // upper bound of sigmoid on output neuron
                    argUseBipolarHiddenNeurons);

            nn.initializeWeights();
            nn.initializeTrainingSet();
            //nn.initializeBias(hiddenBias, outputBias);
            //nn.enableBatchUpdateOption();

            //Activation activation = new TanhActivation();
            //Activation activation = new ReLuActivation();
            //nn.setActivation(activation);

            double error;
            List<Double> errors = new ArrayList<>();

            int epochsToReachTarget = 0;
            boolean targetReached = false;

            String initializedWeights = nn.printHiddenWeights();

            int epochCnt = 0;
            do {
                error = 0.0;
                for (int i = 0; i < NeuralNet.numTrainingSet; i++) {
                    double computedError = nn.train(nn.inputValues[i], nn.actualOutput[i]);
                    error += 0.5*Math.pow(computedError,2);
                }
                errors.add(error);
                if (showErrorAtEachEpoch) System.out.println("--+ Error at epoch " + epochCnt + " is " + error);
                if (showHiddenWeightsAtEachEpoch) System.out.println("--+ Hidden weights at epoch " + epochCnt + " " + nn.printHiddenWeights());

                if (error < target){
                    if (showErrorAtConverge) {
                        System.out.println("Yo!! Error = " + error + " after " + epochCnt + " epochs");
                        System.out.println(initializedWeights);
                    }
                    //output.println("Yo!! Error = " + error + " after " + epochCnt + " epochs");
                    saveToFile(output, errors);
                    epochsToReachTarget = epochCnt;
                    targetReached = true;
                    break;
                }

                epochCnt = epochCnt + 1;
            } while (epochCnt < MAX_EPOCH);

            if (targetReached){
                System.out.println("--+ Target error reached at " + epochsToReachTarget+" epochs");
                return epochCnt;
            }
            else {
                System.out.println("-** Target not reached");
                return DID_NOT_CONVERGE;
            }
        }

        private void saveToFile(PrintWriter output, List<Double> errors) {
            int epochCnt = 0;
            String epochIndexes = "";
            String errorString = "";
            for (Double err : errors) {
                epochIndexes += epochCnt + ",";
                errorString += err + ",";
                epochCnt ++;
            }
            epochIndexes = epochIndexes.substring(0, epochIndexes.length() - 1);
            errorString = errorString.substring(0, errorString.length() - 1);
            output.print(epochIndexes);
            output.println();
            output.print(errorString);
            output.println();
            output.println();
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
            File file = new File(fileName);
            FileWriter writer = new FileWriter(file, true);
            PrintWriter output = new PrintWriter(writer);

            int numCoverages = 0;
            int sum = 0;
            int epochs;
            for (int i=0; i<numTrials; i ++ ){
                NeuralNetRunner myTester = new NeuralNetRunner();
                epochs = myTester.train(output, showErrors.equals("y"), showHiddenWeights.equals("y"), showErrorAtConverge.equals("y"));
                if (epochs != DID_NOT_CONVERGE){
                    numCoverages++;
                    sum += epochs;
                }
            }
            output.close();
            writer.close();

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