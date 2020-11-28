package ece.robocode;

import ece.common.LUTInterface;
import robocode.RobocodeFileOutputStream;

import java.io.*;

public class StateActionLookupTableD4 implements LUTInterface {
    private  int numDim1Levels;
    private  int numDim2Levels;
    private  int numDim3Levels;
    private  int numDim4Levels;

    private double[][][][] lookupTable;
    private int[][][][] visits;

    public StateActionLookupTableD4(
            int numDim1Levels,
            int numDim2Levels,
            int numDim3Levels,
            int numDim4Levels
    ){
        super();
        this.numDim1Levels = numDim1Levels;
        this.numDim2Levels = numDim2Levels;
        this.numDim3Levels = numDim3Levels;
        this.numDim4Levels = numDim4Levels;

        lookupTable = new double[numDim1Levels][numDim2Levels][numDim3Levels][numDim4Levels];
        visits = new int[numDim1Levels][numDim2Levels][numDim3Levels][numDim4Levels];
        this.initialiseLUT();
    }

    @Override
    public void initialiseLUT() {
        for (int a = 0; a <numDim1Levels; a++) {
            for (int b = 0; b <numDim2Levels; b++) {
                for (int c = 0; c < numDim3Levels; c++) {
                    for (int d = 0; d <numDim4Levels; d++) {
                        lookupTable[a][b][c][d] = Math.random();
                            visits[a][b][c][d] = 0;
                    }
                }
            }
        }
    }

    @Override
    public int indexFor(double[] x) {
        return 0;
    }

    @Override
    public double outputFor(double[] x) throws ArrayIndexOutOfBoundsException{
        if (x.length != 4)
            throw new ArrayIndexOutOfBoundsException();
        int a = (int) x[0];
        int b = (int) x[1];
        int c = (int) x[2];
        int d = (int) x[3];

        return lookupTable[a][b][c][d];
    }

    /**
     * using currentReward to compute Q value then update it to Lookup table
     * @param x        The input vector
     * @param target
     * @return
     */
    @Override
    public double train(double[] x, double target) throws ArrayIndexOutOfBoundsException{
        if (x.length != 4)
            throw new ArrayIndexOutOfBoundsException();
        int a = (int) x[0];
        int b = (int) x[1];
        int c = (int) x[2];
        int d = (int) x[3];
        lookupTable[a][b][c][d] = target;
        visits[a][b][c][d]++;
        return 0;
    }

    @Override
    public void save(File argFile) {
        PrintStream saveFile = null;

        try {
            saveFile = new PrintStream( new RobocodeFileOutputStream( argFile ));
        }
        catch (IOException e) {
            System.out.println( "*** Could not create output stream for NN save file.");
        }

        // First line is the number of rows of data
        saveFile.println(numDim1Levels * numDim2Levels * numDim3Levels * numDim4Levels);

        // Second line is the number of dimensions per row
        saveFile.println(5);

        for (int a = 0; a < numDim1Levels; a++) {
            for (int b = 0; b < numDim2Levels; b++) {
                for (int c = 0; c < numDim3Levels; c++) {
                    for (int d = 0; d < numDim4Levels; d++) {
                        // e, d, e2, d2, a, q, visits
                        String row = String.format("%d,%d,%d,%d,%2.5f,%d",
                                a, b, c, d,
                                lookupTable[a][b][c][d],
                                visits[a][b][c][d]
                        );
                        saveFile.println(row);
                    }
                }
            }
        }
        saveFile.close();
    }

    @Override
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
