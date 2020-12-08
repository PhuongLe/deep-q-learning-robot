package ece.robocode;

import ece.backpropagation.StateActionNeuralNet;
import ece.common.*;
import javafx.util.Pair;
import robocode.BattleEndedEvent;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * This robot is cloned from RLRobotV3 which has
 *      3 states: myEnergy, myDistanceToEnemy, my heat gun
 *      4 actions: attack, avoid, runaway, fire
 */
public class NeuralNetworkRobot extends QLearningRobot {
    static private final NeuralNetInterface targetNetwork = new StateActionNeuralNet(false);
    static private final NeuralNetInterface policyNetwork = new StateActionNeuralNet(false);
    static public Experience[] experiences = new Experience[]{};
    static private final int NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS = 100;
    static private final int REPLAY_MEMORY_SIZE = 1; // set it as 1 to experiment one back propagation
    static private final int REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE = 1; //it must be less than REPLAY_MEMORY_SIZE
    static private final int NUM_OBSERVATION = 2; //NUM_OBSERVATION must be greater than REPLAY_MEMORY_SIZE
    static private final int NUM_STOP_ONLINE_TRAINING = 2000;

    static String stateActionNeuralNetWeightsFileName = StateActionNeuralNet.baseFolder + "nn_weights.dat";

    static LogFile logQChangesFile = null;
    static String logQChangesFileName = baseFolder + "-robocode-q-changes.log";
    static LogFile logLossFile = null;
    static String logLossFileName = baseFolder + "-robocode-loss.log";
    static LogFile debugLogFile = null;
    static String debugLogFileName = baseFolder + "-debug.log";

    static double logged_precedingPreviousQValue;
    static double logged_qChange;
    static double logged_loss;
    static double logged_maxQ;
    static double logged_priorQ;

    static StringBuilder debugLog = new StringBuilder();

    private static final Random random = new Random();

    private static boolean enableDebug = false;

    @Override
    protected Pair<String, Double>[] metadata() {
        Pair<String, Double>[] result = super.metadata();
        result = ArrayHelper.push(result, new Pair<>("REPLAY_MEMORY_SIZE", (double) REPLAY_MEMORY_SIZE));
        result = ArrayHelper.push(result, new Pair<>("REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE", (double) REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE));
        result = ArrayHelper.push(result, new Pair<>("NUM_OBSERVATION", (double) NUM_OBSERVATION));
        result = ArrayHelper.push(result, new Pair<>("NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS", (double) NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS));
        result = ArrayHelper.push(result, new Pair<>("NUM_STOP_ONLINE_TRAINING", (double) NUM_STOP_ONLINE_TRAINING));

        return result;
    }

    @Override
    protected void initialize() {
        super.initialize();
        if (logQChangesFile == null) {
            logQChangesFile = new LogFile(getDataFile(logQChangesFileName));
            logQChangesFile.printHyperParameters(this.metadata());

            logLossFile = new LogFile(getDataFile(logLossFileName));
            logLossFile.printHyperParameters(this.metadata());

            if (enableDebug) {
            debugLogFile = new LogFile(getDataFile(debugLogFileName));
                debugLogFile.printHyperParameters(this.metadata());
            }

            try {
                targetNetwork.load(stateActionNeuralNetWeightsFileName);
                policyNetwork.cloneWeights(targetNetwork);
            } catch (IOException e) {
                log.stream.printf("*** Could not initialize neural network from file");
            }
        }
    }

    @Override
    public void onBattleEnded(BattleEndedEvent event) {
        if (enableDebug) {
            debugLogFile.stream.print(debugLog.toString());
        }
    }

    @Override
    protected void trackResults() {
        super.trackResults();

        //track the difference between output Q value with the prior Q value
        logged_qChange = logged_priorQ - logged_precedingPreviousQValue;
        logQChangesFile.stream.printf("%2.10f, %2.10f\n", logged_qChange, logged_precedingPreviousQValue);

        //track the loss between output Q value with the target Q value of the previous state action
        logLossFile.stream.printf("%2.10f\n", logged_loss, logged_maxQ);
    }

    /**
     * Update experiences
     */
    @Override
    protected void prePerformingValueFunctions(){
        if (totalNumRounds > NUM_STOP_ONLINE_TRAINING){
            return;
        }

        //save experience to replay memory
        if (experiences.length >= REPLAY_MEMORY_SIZE){
            if (experiences.length == 1){
                experiences = new Experience[]{};
            }else {
                experiences = ArrayHelper.pop(experiences);
            }
        }
        experiences = ArrayHelper.push(experiences, new Experience(previousState, previousAction, reward, currentState));
    }

    @Override
    protected void postPerformingValueFunctions() throws IOException {
        if (totalNumRounds > NUM_STOP_ONLINE_TRAINING){
            return;
        }

        if (numOfMoves % NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS == 0){
            targetNetwork.cloneWeights(policyNetwork);
        }
        writeDebug("executed action = " + currentAction.name());
    }

    /**
     * Update neural network's weights by doing one back propagation step the loss or previous action state space
     * @param bestActionValue best action value
     */
    @Override
    protected void performValueFunction(Pair<Action.enumActions, Double> bestActionValue) {
        // only train if done observing
        if (totalNumRounds > NUM_STOP_ONLINE_TRAINING || totalNumRounds < NUM_OBSERVATION){
            double[] previousStateAction = previousState.StateActionInputVector(previousAction.ordinal());
            logged_priorQ = policyNetwork.outputFor(previousStateAction);
            return;
        }

        //generate random mini batch state-actions from replay memory
        int[] randomBatchIndexes = new int[REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE];

        //for a shake of easy monitoring robot performance, always put the previous state-action on the random batch
        randomBatchIndexes[0] = REPLAY_MEMORY_SIZE - 1;

        int miniCount = 1;
        while (miniCount < REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE){
            int randomIndex = random.nextInt(REPLAY_MEMORY_SIZE);
            boolean existed = false;
            for (int randomBatchIndex : randomBatchIndexes) {
                if (randomBatchIndex == randomIndex) {
                    existed = true;
                    break;
                }
            }
            if (!existed){
                randomBatchIndexes[miniCount] = randomIndex;
                miniCount++;
            }
        }

        //Sample random batch from replay memory. In this project, it is all reply_memory_size
        for (int i = 0; i < REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE; i++){
            updateWeights(i, randomBatchIndexes, bestActionValue);
        }
    }

    private void updateWeights(int i, int[] randomBatchIndexes, Pair<Action.enumActions, Double> bestActionValue){
        int experienceIndex = randomBatchIndexes[i];
        double[] previousStateAction = experiences[experienceIndex].previousState.StateActionInputVector(experiences[experienceIndex].previousAction.ordinal());

        logged_priorQ = policyNetwork.outputFor(previousStateAction);
        if (i != 0 || (i == 0 && bestActionValue == null)) {
            logged_maxQ = this.getBestAction(experiences[experienceIndex].currentState).getValue();
        }

        logged_loss = ALPHA*(reward + GAMMA*logged_maxQ - logged_priorQ);
        policyNetwork.backwardPropagation(previousStateAction, logged_loss);

        writeDebug("priorQ = " + logged_priorQ);
        writeDebug("maxQ = " + logged_maxQ);
        writeDebug("loss = " + logged_loss);
        writeDebug(policyNetwork.printHiddenWeights());

//        if (i == 0){
//            //track qChange and loss for monitoring robot and neural network performance
//            qChange = priorQ - precedingPreviousQValue;
//            precedingPreviousQValue = priorQ;
//        }
    }

    /**
     * Using neural network to pick the best action for the input state.
     * @param state state
     * @return figure out the best action of the given state
     */
    @Override
    protected Pair<Action.enumActions, Double> getBestAction(State state)
    {
        writeDebug("getting best action for");

        int action = 0;
        double bestQ = -Double.MAX_VALUE;
        //double loss = -Double.MAX_VALUE;
        for(int i = 0; i< Action.NUM_ACTIONS; i++)
        {
            double[] currentStateAction = state.StateActionInputVector(i);
            double computedQValue =  targetNetwork.outputFor(currentStateAction);
            writeDebug("stateAction = " + state.StateActionValueString(i)
                    + ". StateActionBipolar = " + state.StateActionInputVectorString(i)
                    + ". ComputedQValue = " + computedQValue);

            if( computedQValue > bestQ)
            {
                bestQ = computedQValue;
                action = i;
            }
        }
        writeDebug("stateAction = " + state.StateActionValueString(action)
                + ". StateActionBipolar = " + state.StateActionInputVectorString(action)
                + ". bestQ = " + bestQ);
        return new Pair<> (Action.enumActions.values()[action], bestQ);
    }

    private void writeDebug(String message){
        if (!enableDebug){
            return;
        }
        debugLog.append(message + "\n");
    }
}
