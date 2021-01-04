package reinforcement;

import common.*;
import javafx.util.Pair;
import robocode.*;

import java.io.File;
import java.io.IOException;

/**
 * This robot is cloned from RLRobotV3 which has
 *      3 states: myEnergy, myDistanceToEnemy, my heat gun
 *      4 actions: attack, avoid, runaway, fire
 */
public class QLearningRobo extends AdvancedRobot {
    static LogFile log = null;

    public static final boolean useQLearning = true;

    //Discount factor & learning rate used by RL
    protected static double GAMMA = 0.9;
    protected static double ALPHA = 0.4;
    protected static double EPSILON_INITIAL = 0.3;
    protected static int STOP_RANDOM_ACTION_ROUND = 2000;
    protected double epsilon = EPSILON_INITIAL;

    protected double reward = 0.0;
    protected static double accumulativeReward = 0;

    //Rewards
    protected final double badTerminalReward = -10.0;
    protected final double goodTerminalReward = 10.0;

    //set instantReward to 0 to turn off instant rewards
    protected final double badInstantReward = -3.0;
    protected final double goodInstantReward = 3.0;

    static private final int CHUNK_SIZE = 20;
    static protected int numOfMoves = 0;

    static protected State previousState = new State();
    static protected State currentState = new State();
    static protected Action.enumActions previousAction = Action.enumActions.avoid;
    static protected Action.enumActions currentAction = Action.enumActions.avoid;

    protected byte moveDirection = 1;
    static protected int totalNumRounds = 0;
    static protected int totalNumWins = 0;

    static private final LUTInterface q = new QLearningLookupTable(
            State.NUM_ENERGY,
            State.NUM_DISTANCE,
            State.NUM_GUN_HEAT,
            Action.NUM_ACTIONS);

    public void run() {
        initialize();

        //Customize our robot
        setAdjustGunForRobotTurn(true);

        setAdjustRadarForGunTurn(true);

        turnRadarRightRadians(Double.POSITIVE_INFINITY);
    }

    protected void initialize() {
        // Create log file
        if (log == null) {
            log = new LogFile(getDataFile(AppConfiguration.RoboLogFileName));
            log.printHyperParameters(this.metadata());
        }
    }

    /**
     * onScannedRobot: What to do when you see another robot
     */
    public void onScannedRobot(ScannedRobotEvent e) {
        previousState.update(currentState.energy, currentState.distance, currentState.gunHeat, currentState.bearing);
        currentState.update(getEnergy(),
                e.getDistance(),
                getGunHeat(),
                e.getBearing());
        previousAction = currentAction;

        double firePower = Math.min(400 / e.getDistance(), 3);
        //  calculate gun turn toward enemy
        double turn = getHeading() - getGunHeading() + e.getBearing();
        // normalize the turn to take the shortest path there
        setTurnGunRight(normalizeBearing(turn));

        makeDecision();

        executeAction(currentAction, firePower);
        numOfMoves ++;

        setTurnRadarLeftRadians(getRadarTurnRemainingRadians());
    }

    double normalizeBearing(double angle) {
        while (angle >  180) angle -= 360;
        while (angle < -180) angle += 360;
        return angle;
    }

    private void executeAction(Action.enumActions action, double firePower) {
        switch(action){
            case attack:
                attackFace();
                break;
            case fire:
                fire(firePower);
                break;
            case avoid:
                avoid();
                break;
            case runaway:
                runAway();
                break;
        }
    }
    public void avoid()
    {
        // switch directions if we've stopped
        if (getVelocity() == 0)
            moveDirection *= -1;

        // circle our enemy
        setTurnRight(currentState.bearing + 90);
        setAhead(100 * moveDirection);

        if (getTime() % 100 == 0) {
            fire(1);
        }
    }

    public void attackFace()
    {
        setTurnRight(currentState.bearing);
        setAhead(100);

        if (getTime() % 100 == 0) {
            fire(1);
        }

    }

    public void runAway()
    {
        setTurnRight(currentState.bearing - 10);
        setAhead(-100);

        if (getTime() % 100 == 0) {
            fire(1);
        }
    }

    protected void trackResults() {
        //save the current win percentage for each chunk_size
        log.stream.printf("%2.1f\t%2.1f\n", 100.0 * totalNumWins / totalNumRounds, accumulativeReward);
    }

    public void finishOneRound() {
        totalNumRounds++;

        if (((totalNumRounds % CHUNK_SIZE == 0) && totalNumRounds != 0)) {
            //reset accumulative reward after each round
            trackResults();
            accumulativeReward = 0;
        }

        numOfMoves = 0;
    }

    @Override
    public void onDeath(DeathEvent event) {
        reward += badTerminalReward;
        performValueFunction(null);
        finishOneRound();
    }

    @Override
    public void onWin(WinEvent event) {
        totalNumWins++;
        reward += goodTerminalReward;
        performValueFunction(null);
        finishOneRound();
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        double change = event.getBullet().getPower() * goodInstantReward ;
        reward += (int)change;
    }

    @Override
    public void onBulletHitBullet(BulletHitBulletEvent e)
    {
        reward += badInstantReward;
    }
    /**
     * onHitByBullet: What to do when you're hit by a bullet
     */
    public void onHitByBullet(HitByBulletEvent e) {
        double power = e.getBullet().getPower();
        double change = badInstantReward * power;

        reward += (int)change;
    }
    @Override
    public void onHitWall(HitWallEvent event) {
        reward += badInstantReward;
        moveDirection *= -1;
    }

    @Override
    public void onHitRobot(HitRobotEvent event) {
        reward += badInstantReward;
        moveDirection *= -1;
    }

    public void onBulletMissed(BulletMissedEvent e)
    {
        double change = -e.getBullet().getPower();
        reward += (int)change;
    }

    @Override
    public File getDataFile(String filename) {
        return new File(filename);
    }

    @Override
    public void onBattleEnded(BattleEndedEvent event) {
        try {
            q.save(getDataFile(AppConfiguration.RoboLutFileName));
        } catch (IOException e) {
            log.stream.println(e);
        }
    }

    /**
     * Using Lookup table to pick the best action for the input state.
     * @param state the inspecting state
     * @return the best value corresponding to the inspecting state
     */
    protected Pair<Action.enumActions, Double> getBestAction(State state)
    {
        int action = 0;
        double bestQ = -Double.MAX_VALUE;
        for(int i = 0; i< Action.NUM_ACTIONS; i++)
        {
            double[] currentStateAction = state.StateActionValue(i);
            if(q.outputFor(currentStateAction) > bestQ)
            {
                bestQ = q.outputFor(currentStateAction);
                action = i;
            }

        }
        return new Pair<> (Action.enumActions.values()[action], bestQ);
    }


    private double ComputeQWithOffPolicy(State previousState, State currentState
            , Action.enumActions previousAction
            , Pair<Action.enumActions, Double> bestActionValue){
        double[] previousStateAction = previousState.StateActionValue(previousAction.ordinal());
        double priorQ = q.outputFor(previousStateAction);
        double maxQ;
        if (bestActionValue == null) {
            maxQ = getBestAction(currentState).getValue();
        }
        else
        {
            maxQ = bestActionValue.getValue();
        }
        return priorQ + ALPHA*(reward + GAMMA*maxQ - priorQ);
    }

    private double ComputeQWithOnPolicy(State previousState, State currentState, Action.enumActions previousAction, Action.enumActions currentAction){
        double[] previousStateAction = previousState.StateActionValue(previousAction.ordinal());
        double[] currentStateAction = currentState.StateActionValue(currentAction.ordinal());

        double priorQ = q.outputFor(previousStateAction);
        double currentQ = q.outputFor(currentStateAction);

        return priorQ + ALPHA*(reward + GAMMA*currentQ - priorQ);
    }

    protected Pair<String, Double>[] metadata() {
        return new Pair[]{
                new Pair("gamma", GAMMA),
                new Pair("alpha", ALPHA),
                new Pair("epsilon", EPSILON_INITIAL),
                new Pair("STOP_RANDOM_ACTION_ROUND", (double)STOP_RANDOM_ACTION_ROUND),
                new Pair("badInstantReward", badInstantReward),
                new Pair("badTerminalReward", badTerminalReward),
                new Pair("goodInstantReward", goodInstantReward),
                new Pair("goodTerminalReward", goodTerminalReward)
        };
    }

    private void makeDecision() {
        if(numOfMoves == 0) //first move
        {
            currentAction = Action.SelectRandomAction();
            return;
        }
        if (totalNumRounds > STOP_RANDOM_ACTION_ROUND){ //stop exploration after initial rounds
            epsilon = 0.0;
        }

        Pair<Action.enumActions, Double> bestActionValue = null;
        //make decision for next action
        if(Math.random() > epsilon ) //greedy move
        {
            bestActionValue = this.getBestAction(currentState);
            currentAction = bestActionValue.getKey();
        }else{ //try exploration => random move
            currentAction = Action.SelectRandomAction();
        }
        if(currentState.isNotEqual(previousState)) {
            prePerformingValueFunctions();

            //update policy
            performValueFunction(bestActionValue);

            postPerformingValueFunctions();
        }
    }

    protected void prePerformingValueFunctions() {
        accumulativeReward = ALPHA*accumulativeReward + reward;

    }

    protected void postPerformingValueFunctions(){
    }

    /**
     * update Q value by either on or off policy
     * @param bestActionValue the best action
     */
    protected void performValueFunction(Pair<Action.enumActions, Double> bestActionValue) {
        if (useQLearning){
            q.train(previousState.StateActionValue(previousAction.ordinal()),
                    this.ComputeQWithOffPolicy(previousState, currentState, previousAction, bestActionValue));
        }else{
            q.train(previousState.StateActionValue(previousAction.ordinal())
                    , this.ComputeQWithOnPolicy(previousState, currentState, previousAction, currentAction));
        }
        //reset reward
        reward = 0;
    }
}
