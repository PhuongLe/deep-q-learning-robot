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
    static private final NeuralNetInterface targetNetwork = new StateActionNeuralNet();
    static private final NeuralNetInterface policyNetwork = new StateActionNeuralNet();
    static public Experience[] experiences = new Experience[]{};
    static private final int NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS = 100;
    static private final int REPLAY_MEMORY_SIZE = 1; // set it as 1 to experiment one back propagation
    static private final int REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE = 1; //it must be less than REPLAY_MEMORY_SIZE
    static private final int NUM_OBSERVATION = 2; //NUM_OBSERVATION must be greater than REPLAY_MEMORY_SIZE

    static String stateActionNeuralNetWeightsFileName = StateActionNeuralNet.baseFolder + "nn_weights.dat";

    static LogFile logQChangesFile = null;
    static String logQChangesFileName = baseFolder + "-robocode-q-changes.log";
    static LogFile logLossFile = null;
    static String logLossFileName = baseFolder + "-robocode-loss.log";
    static LogFile debugLogFile = null;
    static String debugLogFileName = baseFolder + "-debug.log";

    static double precedingPreviousQValue;
    static double qChange;
    static double loss;

    static StringBuilder debugLog = new StringBuilder();

    private static final Random random = new Random();


    @Override
    protected Pair<String, Double>[] metadata() {
        Pair<String, Double>[] result = super.metadata();
        ArrayHelper.push(result, new Pair<>("REPLAY_MEMORY_SIZE", REPLAY_MEMORY_SIZE));
        ArrayHelper.push(result, new Pair<>("REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE", REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE));
        ArrayHelper.push(result, new Pair<>("NUM_OBSERVATION", NUM_OBSERVATION));
        ArrayHelper.push(result, new Pair<>("NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS", NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS));
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

            debugLogFile = new LogFile(getDataFile(debugLogFileName));
            debugLogFile.printHyperParameters(this.metadata());

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
        debugLogFile.stream.print(debugLog.toString());
    }

    @Override
    protected void trackResults() {
        super.trackResults();

        //track the difference between output Q value with the prior Q value
        logQChangesFile.stream.printf("%2.10f, %2.10f\n", qChange, precedingPreviousQValue);

        //track the loss between output Q value with the target Q value of the previous state action
        logLossFile.stream.printf("%2.10f\n", loss);
    }

    /**
     * Update experiences
     */
    @Override
    protected void prePerformingValueFunctions(){
        //save experience to replay memory
        if (experiences.length >= REPLAY_MEMORY_SIZE){
            ArrayHelper.pop(experiences);
        }
        ArrayHelper.push(experiences, new Experience(previousState, previousAction, reward, currentState));
    }

    @Override
    protected void postPerformingValueFunctions() throws IOException {
        if (numOfMoves % NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS == 0){
            targetNetwork.cloneWeights(policyNetwork);
        }
    }

    /**
     * Update neural network's weights by doing one back propagation step the loss or previous action state space
     * @param bestActionValue best action value
     */
    @Override
    protected void performValueFunction(Pair<Action.enumActions, Double> bestActionValue) {
        // only train if done observing
        if (numOfMoves < NUM_OBSERVATION){
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
            updateWeights(i, randomBatchIndexes);
        }
    }

    private void updateWeights(int i, int[] randomBatchIndexes){
        int experienceIndex = randomBatchIndexes[i];
        double[] previousStateAction = experiences[experienceIndex].previousState.StateActionInputVector(experiences[experienceIndex].previousAction.ordinal());

        double priorQ = policyNetwork.outputFor(previousStateAction);
        double maxQ = this.getBestAction(experiences[experienceIndex].currentState).getValue();

        loss = ALPHA*(reward + GAMMA*maxQ - priorQ);
        policyNetwork.backwardPropagation(previousStateAction, loss);

        debugLog.append("priorQ = " + priorQ + "\n");
        debugLog.append("maxQ = " + maxQ + "\n");
        debugLog.append("loss = " + loss + "\n");

        if (i == 0){
            //track qChange and loss for monitoring robot and neural network performance
            qChange = priorQ - precedingPreviousQValue;
            precedingPreviousQValue = priorQ;
        }
    }

    /**
     * Using neural network to pick the best action for the input state.
     * @param state state
     * @return figure out the best action of the given state
     */
    @Override
    protected Pair<Action.enumActions, Double> getBestAction(State state)
    {
        int action = 0;
        double bestQ = -Double.MAX_VALUE;
        for(int i = 0; i< Action.NUM_ACTIONS; i++)
        {
            double[] currentStateAction = state.StateActionInputVector(i);
            double computedQValue =  targetNetwork.outputFor(currentStateAction);
            if( computedQValue > bestQ)
            {
                bestQ = computedQValue;
                action = i;
            }
        }
        return new Pair<> (Action.enumActions.values()[action], bestQ);
    }
}
