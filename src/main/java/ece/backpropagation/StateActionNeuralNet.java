package ece.backpropagation;

import ece.common.Action;
import ece.common.Experience;
import ece.common.State;
import ece.robocode.StateActionLookupTableD4;

import java.io.*;

//This NeuralNet class is design for a NN of 2+ inputs, 1 hidden layer with 4++ neurons and 1 output
//The number of training set is 4 for each epoch
public class StateActionNeuralNet extends XorNeuralNet {
    public static String baseFolder = "d:\\Google Drive\\LXP\\UBC\\Term 3\\CPEN 502 - ML\\Assignments\\Robocode\\data\\";
    static String lutTableFileName = baseFolder+ "lut.log";

    public static int NUM_TRAINING_SET = State.NUM_ENERGY * State.NUM_DISTANCE * State.NUM_GUN_HEAT * Action.NUM_ACTIONS;
    public static int NUM_INPUTS = State.NUM_DISTANCE + State.NUM_ENERGY + State.NUM_GUN_HEAT + Action.NUM_ACTIONS;
    public static int NUM_HIDDEN = 25;
    public static double LEARNING_RATE = 0.2;
    public static double MOMENTUM = 0.9;

    public static double ARG_A = -1;
    public static double ARG_B = 1;
    public static boolean USE_BIPOLAR = true;

    static private final StateActionLookupTableD4 q = new StateActionLookupTableD4(
            State.NUM_ENERGY,
            State.NUM_DISTANCE,
            State.NUM_GUN_HEAT,
            Action.NUM_ACTIONS);

    public StateActionNeuralNet(){
        super(NUM_INPUTS, NUM_HIDDEN, LEARNING_RATE, MOMENTUM, ARG_A, ARG_B, USE_BIPOLAR);
    }

    @Override
    public void initializeTrainingSet() {
        try{
            q.load(lutTableFileName);
        } catch (IOException e) {
            System.out.println("*** Could not load state action lookup table from file");
            return;
        }
        inputValues = new double[NUM_TRAINING_SET][NUM_INPUTS];
        actualOutput = new double[NUM_TRAINING_SET];

        double qValue;
        int trainingSetIndex = 0;
        double scaleSize = q.GetScaleSize();

        for (int a = 0; a <State.NUM_ENERGY; a++) {
            for (int b = 0; b <State.NUM_DISTANCE; b++) {
                for (int c = 0; c < State.NUM_GUN_HEAT; c++) {
                    for (int d = 0; d <Action.NUM_ACTIONS; d++) {
                        State.enumEnergy energy = State.enumEnergy.values()[a];
                        addEnergyBipolarFormatToDataSet(energy, trainingSetIndex);
                        State.enumDistance distance = State.enumDistance.values()[b];
                        addDistanceBipolarFormatToDataSet(distance, trainingSetIndex);
                        State.enumGunHeat gunHeat = State.enumGunHeat.values()[c];
                        addGunHeatBipolarFormatToDataSet(gunHeat, trainingSetIndex);
                        Action.enumActions action = Action.enumActions.values()[d];
                        addActionBipolarFormatToDataSet(action, trainingSetIndex);

                        qValue = q.outputFor(new double[]{a,b,c,d});
                        actualOutput[trainingSetIndex] = qValue/scaleSize;
                    }
                }
            }
        }
    }

    private void addEnergyBipolarFormatToDataSet(State.enumEnergy energy, int trainingSetIndex) {
        switch (energy){
            case low:
                inputValues[trainingSetIndex][0] = +1;
                inputValues[trainingSetIndex][1] = -1;
                inputValues[trainingSetIndex][2] = -1;
                break;
            case medium:
                inputValues[trainingSetIndex][0] = -1;
                inputValues[trainingSetIndex][1] = 1;
                inputValues[trainingSetIndex][2] = -1;
                break;
            case high:
                inputValues[trainingSetIndex][0] = -1;
                inputValues[trainingSetIndex][1] = -1;
                inputValues[trainingSetIndex][2] = +1;
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + energy);
        }
    }

    private void addDistanceBipolarFormatToDataSet(State.enumDistance distance, int trainingSetIndex) {
        switch (distance){
            case veryClose:
                inputValues[trainingSetIndex][3] = +1;
                inputValues[trainingSetIndex][4] = -1;
                inputValues[trainingSetIndex][5] = -1;
                break;
            case near:
                inputValues[trainingSetIndex][3] = -1;
                inputValues[trainingSetIndex][4] = 1;
                inputValues[trainingSetIndex][5] = -1;
                break;
            case far:
                inputValues[trainingSetIndex][3] = -1;
                inputValues[trainingSetIndex][4] = -1;
                inputValues[trainingSetIndex][5] = +1;
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + distance);
        }
    }

    private void addGunHeatBipolarFormatToDataSet(State.enumGunHeat gunHeat, int trainingSetIndex) {
        switch (gunHeat){
            case low:
                inputValues[trainingSetIndex][6] = +1;
                inputValues[trainingSetIndex][7] = -1;
                break;
            case high:
                inputValues[trainingSetIndex][6] = -1;
                inputValues[trainingSetIndex][7] = 1;
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + gunHeat);
        }
    }

    private void addActionBipolarFormatToDataSet(Action.enumActions action, int trainingSetIndex) {
        switch (action){
            case attack:
                inputValues[trainingSetIndex][8] = +1;
                inputValues[trainingSetIndex][9] = -1;
                inputValues[trainingSetIndex][10] = -1;
                inputValues[trainingSetIndex][11] = -1;
                break;
            case avoid:
                inputValues[trainingSetIndex][8] = -1;
                inputValues[trainingSetIndex][9] = +1;
                inputValues[trainingSetIndex][10] = -1;
                inputValues[trainingSetIndex][11] = -1;
                break;
            case runaway:
                inputValues[trainingSetIndex][8] = -1;
                inputValues[trainingSetIndex][9] = -1;
                inputValues[trainingSetIndex][10] = +1;
                inputValues[trainingSetIndex][11] = -1;
                break;
            case fire:
                inputValues[trainingSetIndex][8] = -1;
                inputValues[trainingSetIndex][9] = -1;
                inputValues[trainingSetIndex][10] = -1;
                inputValues[trainingSetIndex][11] = +1;
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + action);
        }
    }
}
