package ece.backpropagation;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

public class ProcessLuT {

    static String lutFileName = "D:\\Work\\Courses\\Winter2020Term1\\CPEN502ArchitectureForLearningSystems\\Assignment3\\Phoebes_repo\\Matlab_processing\\Process_LuT\\" + "2020-12-04-00-57-05-robocode-lut_preprocess.log";
    static int numDim1Levels = 3;
    static int numDim2Levels = 3;
    static int numDim3Levels = 2;
    static int numDim4Levels = 4;
    private double[][][][] lookupTable;
    private int[][][][] visits;

    // Processed look-up table
    static String processedlutFileName = "D:\\Work\\Courses\\Winter2020Term1\\CPEN502ArchitectureForLearningSystems\\Assignment3\\Phoebes_repo\\Matlab_processing\\Process_LuT\\" + "2020-12-04-00-57-05-robocode-lut_processed.log";
    static int numBitsEnergy = 2;
    static int numBitsDistance = 2;
    static int numBitsGunHeat = 1;
    static int numBitsAction = 3;
    public double[][] stateActionPatterns;
    public double[] stateActionPatternsExpectedOutput;

    // Constructor
    public ProcessLuT(){

//        // Initialize LuT
//        lookupTable = new double[numDim1Levels][numDim2Levels][numDim3Levels][numDim4Levels];
//        visits = new int[numDim1Levels][numDim2Levels][numDim3Levels][numDim4Levels];

        // Initialize processed LuT (LuT processed offline in Matlab and loaded from file)
        stateActionPatterns = new double[numDim1Levels*numDim2Levels*numDim3Levels*numDim4Levels][numBitsEnergy+numBitsDistance+numBitsGunHeat+numBitsAction];
        stateActionPatternsExpectedOutput = new double[numDim1Levels*numDim2Levels*numDim3Levels*numDim4Levels];
    }

    public void load_file() throws IOException {

//        load(lutFileName);
        loadProcessedLookUpTable(processedlutFileName);
    }

    public void loadProcessedLookUpTable(String argFileName) throws IOException {
        FileInputStream inputFile = new FileInputStream( argFileName );
        BufferedReader inputReader = new BufferedReader(new InputStreamReader( inputFile ));
        int numExpectedRows = numDim1Levels * numDim2Levels * numDim3Levels * numDim4Levels;

        // Check the number of rows is compatible
        int numRows = Integer.valueOf( inputReader.readLine() );
        // Check the number of dimensions is compatible
        int numDimensions = Integer.valueOf( inputReader.readLine() );

        if (numRows != numExpectedRows || numDimensions != numBitsEnergy+numBitsDistance+numBitsGunHeat+numBitsAction) {
            System.out.printf (
                    "*** rows/dimensions expected is %s/%s but %s/%s encountered\n",
                    numExpectedRows, numBitsEnergy+numBitsDistance+numBitsGunHeat+numBitsAction, numRows, numDimensions
            );
            inputReader.close();
            throw new IOException();
        }

        for(int indRow = 0; indRow < numRows; indRow++){
            String line = inputReader.readLine();
            String tokens[] = line.split(",");
            for(int indCol = 0; indCol < numDimensions; indCol++){
                double stateActionRowItem = Double.parseDouble(tokens[indCol]);
                stateActionPatterns[indRow][indCol] = stateActionRowItem;
            }
            double QRowItem = Double.parseDouble(tokens[8]);
            stateActionPatternsExpectedOutput[indRow] = QRowItem;
        }
        inputReader.close();

    }

    public void load(String argFileName) throws IOException {
        FileInputStream inputFile = new FileInputStream( argFileName );
        BufferedReader inputReader = new BufferedReader(new InputStreamReader( inputFile ));
        //int numExpectedRows = numDim1Levels * numDim2Levels * numDim3Levels * numDim4Levels * numDim5Levels;
        int numExpectedRows = numDim1Levels * numDim2Levels * numDim3Levels * numDim4Levels;

        // Check the number of rows is compatible
        int numRows = Integer.valueOf( inputReader.readLine() );
        // Check the number of dimensions is compatible
        int numDimensions = Integer.valueOf( inputReader.readLine() );

        if ( numRows != numExpectedRows || numDimensions != 4) {
            System.out.printf (
                    "*** rows/dimensions expected is %s/%s but %s/%s encountered\n",
                    numExpectedRows, 4, numRows, numDimensions
            );
            inputReader.close();
            throw new IOException();
        }

        for (int a = 0; a < numDim1Levels; a++) {
            for (int b = 0; b < numDim2Levels; b++) {
                for (int c = 0; c < numDim3Levels; c++) {
                    for (int d = 0; d < numDim4Levels; d++) {
                        // Read line formatted like this: <e,d,e2,d2,a,q,visits\n>
                        String line = inputReader.readLine();
                        String tokens[] = line.split(",");
                        double q = Double.parseDouble(tokens[4]);
                        int v = Integer.parseInt(tokens[5]);
                        lookupTable[a][b][c][d] = q;
                        visits[a][b][c][d] = v;

                    }
                }
            }
        }
        inputReader.close();
    }

}
