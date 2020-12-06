package ece.robocode;

import ece.backpropagation.StateActionNeuralNet;
import ece.common.*;
import javafx.util.Pair;

import java.io.IOException;
import java.util.Random;

/**
 * This robot is cloned from RLRobotV3 which has
 *      3 states: myEnergy, myDistanceToEnemy, my heat gun
 *      4 actions: attack, avoid, runaway, fire
 */
public class NeuralNetworkRobot extends QFunctionRobot {
    static private NeuralNetInterface targetNetwork;
    static private NeuralNetInterface policyNetwork;
    public static Experience[] experiences;
    private static int NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS = 100;
    private static int REPLAY_MEMORY_SIZE = 1; // set it as 1 to experiment one back propagation
    private static int REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE = 1; //it must be less than REPLAY_MEMORY_SIZE
    private static int NUM_OBSERVATION = 2; //NUM_OBSERVATION must be greater than REPLAY_MEMORY_SIZE

    static LogFile logQChangesFile = null;
    static String logQChangesFileName = baseFolder+ "-robocode-q-changes.log";
    static LogFile logLossFile = null;
    static String logLossFileName = baseFolder+ "-robocode-loss.log";

    double precedingPreviousQValue;
    double qChange;
    double loss;

    private static Random random = new Random();

    @Override
    protected void initializeLog() {
        super.initializeLog();
        if (logQChangesFile == null) {
            logQChangesFile = new LogFile(getDataFile(logQChangesFileName));
            logQChangesFile.printHyperParameters(this.metadata());
        }
        if (logLossFile == null) {
            logLossFile = new LogFile(getDataFile(logLossFileName));
            logLossFile.printHyperParameters(this.metadata());
        }
    }

    @Override
    protected void trackResults() {
        super.trackResults();

        //track the difference between output Q value with the prior Q value
        logQChangesFile.stream.printf("%2.1f\n", 100.0 * qChange);

        //track the loss between output Q value with the target Q value of the previous state action
        logLossFile.stream.printf("%2.1f\n", 100.0 * loss);
    }

    @Override
    protected void initializeQLearning() throws IOException {
        experiences = new Experience[]{};
        targetNetwork = new StateActionNeuralNet();
        policyNetwork = new StateActionNeuralNet();
        String stateActionNeuralNetWeightsFileName = StateActionNeuralNet.baseFolder + "nn_weights.dat";
        targetNetwork.load(stateActionNeuralNetWeightsFileName);
        policyNetwork.cloneWeights(targetNetwork);
    }

    /**
     * Update experiences
     * @throws IOException
     */
    @Override
    protected void prePerformingValueFunctions() throws IOException {
        //save experience to replay memory
        if (experiences.length >= REPLAY_MEMORY_SIZE){
            ArrayHelper.pop(experiences);
        }
        ArrayHelper.push(experiences, new Experience(this.previousState, this.previousAction, this.reward, this.currentState));
    }

    @Override
    protected void postPerformingValueFunctions() throws IOException {
        if (numOfMoves % NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS == 0){
            policyNetwork.cloneWeights(targetNetwork);
        }
    }

    /**
     * Update neural network's weights by doing one back propagation step the loss or previous action state space
     * @param bestActionValue
     */
    @Override
    protected void performValueFunction(Pair<Action.enumActions, Double> bestActionValue) {
        // only train if done observing
        if (numOfMoves < NUM_OBSERVATION){
            return;
        }

        //generate random mini batch state-actions from replay memory
        int[] randomBatchIndexs = new int[REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE];

        //for a shake of easy monitoring robot performance, always put the previous state-action on the random batch
        randomBatchIndexs[0] = REPLAY_MEMORY_SIZE - 1;

        int miniCount = 1;
        while (miniCount < REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE){
            int randomIndex = random.nextInt(REPLAY_MEMORY_SIZE);
            boolean existed = false;
            for (int i = 0; i <randomBatchIndexs.length; i++){
                if (randomBatchIndexs[i] == randomIndex){
                    existed = true;
                    break;
                }
            }
            if (!existed){
                randomBatchIndexs[miniCount] = randomIndex;
                miniCount++;
            }
        }

        //Sample random batch from replay memory. In this project, it is all reply_memory_size
        for (int i = 0; i< randomBatchIndexs.length; i++){
            int experienceIndex = randomBatchIndexs[i];
            double[] previousStateAction = experiences[experienceIndex].previousState.StateActionValue(experiences[experienceIndex].previousAction.ordinal());

            double priorQ = policyNetwork.outputFor(previousStateAction);
            double maxQ = this.getBestAction(experiences[experienceIndex].currentState).getValue();

            loss = ALPHA*(reward + GAMMA*maxQ - priorQ);
            policyNetwork.backwardPropagation(previousStateAction, loss);
            
            //track qChange and loss for monitoring robot and neural network performance
            if (i == randomBatchIndexs.length - 1){
                qChange = priorQ - precedingPreviousQValue;
                precedingPreviousQValue = priorQ;
            }
        }
    }

    /**
     * Using neural network to pick the best action for the input state.
     * @param state
     * @return
     */
    @Override
    protected Pair<Action.enumActions, Double> getBestAction(State state)
    {
        int action = 0;
        double bestQ = -Double.MAX_VALUE;
        for(int i = 0; i< Action.NUM_ACTIONS; i++)
        {
            double[] currentStateAction = state.StateActionValue(i);
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
