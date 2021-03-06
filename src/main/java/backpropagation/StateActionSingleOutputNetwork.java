package backpropagation;

import common.*;
import reinforcement.QLearningLookupTable;

import java.io.*;

//This NeuralNet class is design for a NN of 2+ inputs, 1 hidden layer with 4++ neurons and 1 output
//The number of training set is 4 for each epoch
public class StateActionSingleOutputNetwork extends BackpropagationBaseNetwork {


    public static int NUM_TRAINING_SET = State.NUM_ENERGY * State.NUM_DISTANCE * State.NUM_GUN_HEAT * Action.NUM_ACTIONS;
    public static int NUM_INPUTS = State.NUM_DISTANCE + State.NUM_ENERGY + State.NUM_GUN_HEAT + Action.NUM_ACTIONS;
    public static int NUM_OUTPUTS = 1;
    public static int NUM_HIDDEN = 12;
    public static double LEARNING_RATE = 0.01;
    public static double MOMENTUM = 0.9;

    public static double ARG_A = -1;
    public static double ARG_B = 1;
    public static boolean USE_BIPOLAR = true;

    static private final LUTInterface q = new QLearningLookupTable(
            State.NUM_ENERGY,
            State.NUM_DISTANCE,
            State.NUM_GUN_HEAT,
            Action.NUM_ACTIONS);

    public StateActionSingleOutputNetwork(boolean initializeTrainingSet){
        super(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS, LEARNING_RATE, MOMENTUM, ARG_A, ARG_B, USE_BIPOLAR, initializeTrainingSet);
    }

    @Override
    public void initializeTrainingSet() {
        try{
            q.load(AppConfiguration.ReferencedLutTableFileName);
        } catch (IOException e) {
            System.out.println("*** Could not load state action lookup table from file");
            return;
        }
        inputValues = new double[NUM_TRAINING_SET][NUM_INPUTS];
        actualOutputs = new double[NUM_TRAINING_SET][NUM_OUTPUTS];

        double qValue;
        int trainingSetIndex = 0;
        double scaleSize = q.GetScaleSize();

        for (int a = 0; a <State.NUM_ENERGY; a++) {
            for (int b = 0; b <State.NUM_DISTANCE; b++) {
                for (int c = 0; c < State.NUM_GUN_HEAT; c++) {
                    for (int d = 0; d <Action.NUM_ACTIONS; d++) {
                        qValue = q.outputFor(new double[]{a,b,c,d});
                        if (qValue == Double.MAX_VALUE){
                            continue;
                        }
                        State.enumEnergy energy = State.enumEnergy.values()[a];
                        addEnergyBipolarFormatToDataSet(energy, trainingSetIndex);
                        State.enumDistance distance = State.enumDistance.values()[b];
                        addDistanceBipolarFormatToDataSet(distance, trainingSetIndex);
                        State.enumGunHeat gunHeat = State.enumGunHeat.values()[c];
                        addGunHeatBipolarFormatToDataSet(gunHeat, trainingSetIndex);
                        Action.enumActions action = Action.enumActions.values()[d];
                        addActionBipolarFormatToDataSet(action, trainingSetIndex);

                        actualOutputs[trainingSetIndex][0] = qValue/scaleSize;
                        trainingSetIndex++;
                    }
                }
            }
        }
        NUM_TRAINING_SET = trainingSetIndex;
    }

    public static double[] MapStateActionToInputVector(State.enumEnergy energy, State.enumDistance distance, State.enumGunHeat gunHeat, Action.enumActions action){
        double[] results = new double[NUM_INPUTS];

        double[] energyBipolar = mapEnergyToBipolarFormat(energy);
        for (int i = 0; i< energyBipolar.length; i++){
            results[i] = energyBipolar[i];
        }

        double[] distanceBipolar = mapDistanceToBipolarFormat(distance);
        for (int i = 0; i< distanceBipolar.length; i++){
            results[i + State.NUM_ENERGY] = distanceBipolar[i];
        }

        double[] gunHeatBipolar = mapGunHeatToBipolarFormat(gunHeat);
        for (int i = 0; i< gunHeatBipolar.length; i++){
            results[i + State.NUM_ENERGY + State.NUM_DISTANCE] = gunHeatBipolar[i];
        }

        double[] actionBipolar = mapActionToBipolarFormat(action);
        for (int i = 0; i< actionBipolar.length; i++){
            results[i + State.NUM_ENERGY + State.NUM_DISTANCE + State.NUM_GUN_HEAT] = actionBipolar[i];
        }
        return results;
    }

    private void addEnergyBipolarFormatToDataSet(State.enumEnergy energy, int trainingSetIndex) {
        double[] inputVector = mapEnergyToBipolarFormat(energy);
        for (int i = 0; i< inputVector.length; i++){
            inputValues[trainingSetIndex][i] = inputVector[i];
        }
    }

    private void addDistanceBipolarFormatToDataSet(State.enumDistance distance, int trainingSetIndex) {
        double[] inputVector = mapDistanceToBipolarFormat(distance);
        for (int i = 0; i< inputVector.length; i++){
            inputValues[trainingSetIndex][i + State.NUM_ENERGY] = inputVector[i];
        }
    }

    private void addGunHeatBipolarFormatToDataSet(State.enumGunHeat gunHeat, int trainingSetIndex) {
        double[] inputVector = mapGunHeatToBipolarFormat(gunHeat);
        for (int i = 0; i< inputVector.length; i++){
            inputValues[trainingSetIndex][i + State.NUM_ENERGY + State.NUM_DISTANCE] = inputVector[i];
        }
    }

    private void addActionBipolarFormatToDataSet(Action.enumActions action, int trainingSetIndex) {
        double[] inputVector = mapActionToBipolarFormat(action);
        for (int i = 0; i< inputVector.length; i++){
            inputValues[trainingSetIndex][i + State.NUM_ENERGY + State.NUM_DISTANCE + State.NUM_GUN_HEAT] = inputVector[i];
        }
    }

    private static double[] mapEnergyToBipolarFormat(State.enumEnergy energy){
        double[] result = new double[State.NUM_ENERGY];
        switch (energy){
            case low:
                result[0] = +1;
                result[1] = -1;
                result[2] = -1;
                break;
            case medium:
                result[0] = -1;
                result[1] = 1;
                result[2] = -1;
                break;
            case high:
                result[0] = -1;
                result[1] = -1;
                result[2] = +1;
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + energy);
        }
        return result;
    }

    private static double[] mapDistanceToBipolarFormat(State.enumDistance distance){
        double[] result = new double[State.NUM_DISTANCE];
        switch (distance){
            case veryClose:
                result[0] = +1;
                result[1] = -1;
                result[2] = -1;
                break;
            case near:
                result[0] = -1;
                result[1] = 1;
                result[2] = -1;
                break;
            case far:
                result[0] = -1;
                result[1] = -1;
                result[2] = +1;
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + distance);
        }
        return result;
    }

    private static double[] mapGunHeatToBipolarFormat(State.enumGunHeat gunHeat){
        double[] result = new double[State.NUM_GUN_HEAT];
        switch (gunHeat){
            case low:
                result[0] = +1;
                result[1] = -1;
                break;
            case high:
                result[0] = -1;
                result[1] = 1;
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + gunHeat);
        }
        return result;
    }

    private static double[] mapActionToBipolarFormat(Action.enumActions action){
        double[] result = new double[Action.NUM_ACTIONS];
        switch (action){
            case attack:
                result[0] = +1;
                result[1] = -1;
                result[2] = -1;
                result[3] = -1;
                break;
            case avoid:
                result[0] = -1;
                result[1] = +1;
                result[2] = -1;
                result[3] = -1;
                break;
            case runaway:
                result[0] = -1;
                result[1] = -1;
                result[2] = +1;
                result[3] = -1;
                break;
            case fire:
                result[0] = -1;
                result[1] = -1;
                result[2] = -1;
                result[3] = +1;
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + action);
        }
        return result;
    }


}
