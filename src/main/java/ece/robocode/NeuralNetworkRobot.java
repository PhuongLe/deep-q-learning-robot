package ece.robocode;

import ece.backpropagation.StateActionNeuralNet;
import ece.common.*;
import javafx.util.Pair;

import java.io.IOException;

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
    private static int REPLAY_MEMORY_SIZE = 5;

    @Override
    protected void initializeQLearning() throws IOException {
        experiences = new Experience[]{};
        targetNetwork = new StateActionNeuralNet();
        policyNetwork = new StateActionNeuralNet();
        String stateActionNeuralNetWeightsFileName = StateActionNeuralNet.baseFolder + "nn_weights.dat";
        targetNetwork.load(stateActionNeuralNetWeightsFileName);
        policyNetwork.cloneWeights(targetNetwork);
    }

    @Override
    protected void prePerformingValueFunctions() throws IOException {
        //save experience to replay memory
        if (experiences.length >= REPLAY_MEMORY_SIZE){
            ArrayHelper.pop(experiences);
        }
        ArrayHelper.push(experiences, new Experience(this.previousState, this.previousAction, this.reward, this.currentState));

        //Sample random batch from replay memory. In this project, it is all reply_memory_size

        //Preprocess states from batch

        //Pass batch of preprocessed states to policy network.

        //Calculate loss between output Q-values and target Q-values.
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
        //First, calculate loss between output Q-value of prior state action and target Q-value (max Q-value) of current state

        double[] previousStateAction = previousState.StateActionValue(previousAction.ordinal());

        //Q-value of prior state action
        double priorQ = targetNetwork.outputFor(previousStateAction);

        double maxQ;
        if (bestActionValue == null) {
            maxQ = GetBestAction(currentState).getValue();
        }
        else
        {
            maxQ = bestActionValue.getValue();
        }

        double loss = ALPHA*(reward + GAMMA*maxQ - priorQ);

        //Second, update weights in the policy network to minimize lost
        targetNetwork.backwardPropagation(previousStateAction, loss);
    }

    /**
     * Using neural network to pick the best action for the input state.
     * @param state
     * @return
     */
    @Override
    protected Pair<Action.enumActions, Double> GetBestAction(State state)
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
