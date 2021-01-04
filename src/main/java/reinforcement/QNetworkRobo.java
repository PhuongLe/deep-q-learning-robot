package reinforcement;

import backpropagation.StateActionSingleOutputNetwork;
import common.*;
import javafx.util.Pair;
import robocode.BattleEndedEvent;

import java.io.IOException;
import java.util.Random;

/**
 * This robot is cloned from RLRobotV3 which has
 *      3 states: myEnergy, myDistanceToEnemy, my heat gun
 *      4 actions: attack, avoid, runaway, fire
 */
public class QNetworkRobo extends QLearningRobo {
    static private final NeuralNetInterface targetNetwork = new StateActionSingleOutputNetwork(false);
    static private final NeuralNetInterface policyNetwork = new StateActionSingleOutputNetwork(false);
    static public Experience[] experiences = new Experience[]{};
    static private final int NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS = 100;
    static private final int REPLAY_MEMORY_SIZE = 100; // set it as 1 to experiment one back propagation
    static private final int REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE = 50; //it must be less than REPLAY_MEMORY_SIZE
    static private final int NUM_OBSERVATION = 0; //NUM_OBSERVATION must be greater than REPLAY_MEMORY_SIZE
    static private final int NUM_STOP_ONLINE_TRAINING = 20000;

    static String stateActionNeuralNetWeightsFileName = AppConfiguration.PretrainedNetworkFileName;

    static State plottedState1 = new State(State.enumEnergy.high, State.enumDistance.far, State.enumGunHeat.high);
    static State plottedState2 = new State(State.enumEnergy.high, State.enumDistance.near, State.enumGunHeat.high){};
    static State plottedState3 = new State(State.enumEnergy.medium, State.enumDistance.near, State.enumGunHeat.low){};
    static Action.enumActions plottedAction1 = Action.enumActions.avoid;
    static Action.enumActions plottedAction2 = Action.enumActions.fire;
    static Action.enumActions plottedAction3 = Action.enumActions.runaway;
    static double plotted_previousQValue1;
    static double plotted_previousQValue2;
    static double plotted_previousQValue3;
    static double plotted_previousError1 = Double.MAX_VALUE;
    static double plotted_previousError2 = Double.MAX_VALUE;
    static double plotted_previousError3 = Double.MAX_VALUE;
    static double plotted_totalErrors;
    static double new_plotted_totalErrors;

    static LogFile logQChangesFile1 = null;
    static String logQChangesFileName1 = AppConfiguration.FilePrefix + "-robocode-q-change1.log";
    static LogFile logQChangesFile2 = null;
    static String logQChangesFileName2 = AppConfiguration.FilePrefix + "-robocode-q-change2.log";
    static LogFile logQChangesFile3 = null;
    static String logQChangesFileName3 = AppConfiguration.FilePrefix + "-robocode-q-change3.log";
    static LogFile logLossFile = null;
    static String logLossFileName = AppConfiguration.FilePrefix + "-robocode-loss.log";
    static LogFile debugLogFile = null;
    static String debugLogFileName = AppConfiguration.FilePrefix + "-debug.log";


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
        if (logLossFile == null) {
            logQChangesFile1 = new LogFile(getDataFile(logQChangesFileName1));
            logQChangesFile1.printHyperParameters(this.metadata());

            logQChangesFile2 = new LogFile(getDataFile(logQChangesFileName2));
            logQChangesFile2.printHyperParameters(this.metadata());

            logQChangesFile3 = new LogFile(getDataFile(logQChangesFileName3));
            logQChangesFile3.printHyperParameters(this.metadata());

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
        super.onBattleEnded(event);
        if (enableDebug) {
            debugLogFile.stream.print(debugLog.toString());
        }
    }

    @Override
    protected void trackResults() {
        super.trackResults();

        //track the loss between output Q value with the target Q value of the previous state action
        //logLossFile.stream.printf("%2.10f\n", logged_loss);
    }

    /**
     * Update experiences
     */
    @Override
    protected void prePerformingValueFunctions(){
        super.prePerformingValueFunctions();

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
    protected void postPerformingValueFunctions() {
        if (totalNumRounds > NUM_STOP_ONLINE_TRAINING){
            return;
        }

        if (totalNumRounds % NUM_TIMES_TO_SYNC_VALUE_FUNCTIONS == 0){
            targetNetwork.cloneWeights(policyNetwork);
            writeDebug(policyNetwork.printAllWeights());
        }
        writeDebug("executed action = " + currentAction.name());
    }

    /**
     * Update neural network's weights by doing one back propagation step the loss or previous action state space
     */
    @Override
    protected void performValueFunction(Pair<Action.enumActions, Double> bestActionValue) {
        // only train if done observing
        if (totalNumRounds > NUM_STOP_ONLINE_TRAINING || totalNumRounds < NUM_OBSERVATION){
//            double[] previousStateAction = previousState.StateActionInputVector(previousAction.ordinal());
//            logged_priorQ = policyNetwork.outputFor(previousStateAction);
            return;
        }

        //generate random mini batch state-actions from replay memory
        int[] randomBatchIndexes = new int[REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE];
        for (int a = 0; a <REPLAY_MEMORY_RANDOM_MINI_BATCH_SIZE; a++) {
            randomBatchIndexes[a] = -1;
        }
        //for a shake of easy monitoring robot performance, always put the previous state-action on the random batch
        //randomBatchIndexes[0] = REPLAY_MEMORY_SIZE - 1;

        int miniCount = 0;
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

    private double[] convertToStateActionVector(State state, Action.enumActions action){
        return StateActionSingleOutputNetwork.MapStateActionToInputVector(state.energy, state.distance, state.gunHeat, action);
    }

    private void updateWeights(int i, int[] randomBatchIndexes){
        int experienceIndex = randomBatchIndexes[i];
        double[] previousStateAction = convertToStateActionVector(experiences[experienceIndex].previousState,
                                                                  experiences[experienceIndex].previousAction);

        double priorQ = policyNetwork.outputFor(previousStateAction);

        double maxQ = this.getBestAction(experiences[experienceIndex].currentState).getValue();

        double loss = ALPHA*(experiences[experienceIndex].currentReward/137 + GAMMA*maxQ - priorQ);
        policyNetwork.train(previousStateAction, loss);
        //updatePlottedQChanges(priorQ);
        updateErrorChanges(Math.abs(loss)/2);

//        writeDebug("priorQ = " + priorQ);
//        writeDebug("maxQ = " + maxQ);
//        writeDebug("loss = " + loss);
//        if (i == 0){
//            //track qChange and loss for monitoring robot and neural network performance
//            logged_priorQ = priorQ;
//            logged_loss = loss;
//            logged_maxQ = maxQ;
//            writeDebug("logged_maxQ = " + logged_maxQ);
//            writeDebug("logged_loss = " + logged_loss);
//            //writeDebug(policyNetwork.printAllWeights());
//        }
    }
    private void updateErrorChanges(double loss) {
        if (!previousState.isNotEqual(plottedState1) && previousAction.equals(plottedAction1)){
            if (plotted_previousError1 == Double.MAX_VALUE){
                logLossFile.stream.printf("start plotting state-action space 1\n");
                plotted_previousError1 = 0;
            }

            new_plotted_totalErrors = plotted_totalErrors - plotted_previousError1 + loss;
            plotted_previousError1 = loss;
        }
        if (!previousState.isNotEqual(plottedState2) && previousAction.equals(plottedAction2)){
            if (plotted_previousError2 == Double.MAX_VALUE){
                logLossFile.stream.printf("start plotting state-action space 2\n");
                plotted_previousError2 = 0;
            }
            new_plotted_totalErrors = plotted_totalErrors - plotted_previousError2 + loss;
            plotted_previousError2 = loss;
        }
        if (!previousState.isNotEqual(plottedState3) && previousAction.equals(plottedAction3)){
            if (plotted_previousError3 == Double.MAX_VALUE){
                logLossFile.stream.printf("start plotting state-action space 3\n");
                plotted_previousError3 = 0;
            }
            new_plotted_totalErrors = plotted_totalErrors - plotted_previousError3 + loss;
            plotted_previousError3 = loss;
        }

        if ((totalNumRounds % 20 == 0 && totalNumRounds >0) && (new_plotted_totalErrors != plotted_totalErrors)){
            logLossFile.stream.printf("%2.10f\n", new_plotted_totalErrors);
        }
        plotted_totalErrors = new_plotted_totalErrors;
    }
    private void updatePlottedQChanges(double priorQ) {
        if (!previousState.isNotEqual(plottedState1) && previousAction.equals(plottedAction1)){
            double change = Math.abs(priorQ - plotted_previousQValue1);
            //ArrayHelper.push(plotted_qChange1, change);
            if (((totalNumRounds % 20 == 0) && totalNumRounds != 0)) {
                logQChangesFile1.stream.printf("%2.10f\n", change);
            }
            plotted_previousQValue1 = priorQ;
            return;
        }
        if (!previousState.isNotEqual(plottedState2) && previousAction.equals(plottedAction2)){
            double change = Math.abs(priorQ - plotted_previousQValue2);
            if (((totalNumRounds % 20 == 0) && totalNumRounds != 0)) {
                logQChangesFile2.stream.printf("%2.10f\n", change);
            }
            plotted_previousQValue2 = priorQ;
            return;
        }
        if (!previousState.isNotEqual(plottedState3) && previousAction.equals(plottedAction3)){
            double change = Math.abs(priorQ - plotted_previousQValue3);
            if (((totalNumRounds % 20 == 0) && totalNumRounds != 0)) {
                logQChangesFile3.stream.printf("%2.10f\n", change);
            }
            plotted_previousQValue3 = priorQ;
            return;
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
        writeDebug("getting best action for");

        int action = 0;
        double bestQ = -Double.MAX_VALUE;
        //double loss = -Double.MAX_VALUE;
        for(int i = 0; i< Action.NUM_ACTIONS; i++)
        {
            double[] currentStateAction = convertToStateActionVector(state, Action.enumActions.values()[i]);
            double computedQValue =  targetNetwork.outputFor(currentStateAction);
//            writeDebug("stateAction = " + state.StateActionValueString(i)
//                    + ". StateActionBipolar = " + state.StateActionInputVectorString(i)
//                    + ". ComputedQValue = " + computedQValue);

            if( computedQValue > bestQ)
            {
                bestQ = computedQValue;
                action = i;
            }
        }

        if (enableDebug) {
            writeDebug("stateAction = " + state.StateActionValueString(action)
                    + ". StateActionBipolar = " + state.StateActionInputVectorString(convertToStateActionVector(state, Action.enumActions.values()[action]))
                    + ". bestQ = " + bestQ);
        }

        return new Pair<> (Action.enumActions.values()[action], bestQ);
    }

    private void writeDebug(String message){
        if (!enableDebug){
            return;
        }
        debugLog.append(message + "\n");
    }
}
