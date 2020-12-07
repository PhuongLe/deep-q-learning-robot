package ece.robocode;

import ece.common.Action;
import ece.common.LogFile;
import ece.common.State;
import javafx.util.Pair;
import robocode.*;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * This robot is cloned from RLRobotV3 which has
 *      3 states: myEnergy, myDistanceToEnemy, my heat gun
 *      4 actions: attack, avoid, runaway, fire
 */
public class QLearningRobot extends AdvancedRobot {
    static LogFile log = null;
    static String baseFolder = "d:\\Google Drive\\LXP\\UBC\\Term 3\\CPEN 502 - ML\\Assignments\\Robocode\\out\\report\\" + new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss") .format(new Date());
    static String logFileName = baseFolder+ "-robocode.log";
    static String lutFileName = baseFolder+ "-robocode-lut.log";


    public static final boolean useQLearning = true;

    //Discount factor & learning rate used by RL
    protected static double GAMMA = 0.7;
    protected static double ALPHA = 0.5;
    protected static double EPSILON_INITIAL = 0.3;
    protected static int STOP_RANDOM_ACTION_ROUND = 1000;
    protected double epsilon = EPSILON_INITIAL;

    protected double reward = 0.0;

    //Rewards
    protected final double badTerminalReward = -10.0;
    protected final double goodTerminalReward = 10.0;

    //set instantReward to 0 to turn off instant rewards
    protected final double badInstantReward = -3.0;
    protected final double goodInstantReward = 3.0;

    static private final int CHUNK_SIZE = 20;
    static protected int numOfMoves = 0;

    //static int previousState=0, currentState=0;
    static protected State previousState = new State();
    static protected State currentState = new State();
    static protected Action.enumActions previousAction = Action.enumActions.avoid;
    static protected Action.enumActions currentAction = Action.enumActions.avoid;

    protected byte moveDirection = 1;
    static protected int totalNumRounds = 0;
    static protected int totalNumWins = 0;

    static private final StateActionLookupTableD4 q = new StateActionLookupTableD4(
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
            log = new LogFile(getDataFile(logFileName));
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

        try {
            makeDecision();
        } catch (IOException ioException) {
            ioException.printStackTrace();
        }

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

    @Override
    public void onBulletHit(BulletHitEvent event) {
        //currentReward = goodInstantReward;
        double change = event.getBullet().getPower() * goodInstantReward ;
        //out.println("Bullet Hit: " + change);
        reward += (int)change;
    }

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
    public void onDeath(DeathEvent event) {
        reward += badTerminalReward;
    }

    @Override
    public void onRoundEnded(RoundEndedEvent event) {
        super.onRoundEnded(event);
        totalNumRounds++;
        //log.stream.println("total number rounds " + totalNumRounds);

        if (((totalNumRounds % CHUNK_SIZE == 0) && totalNumRounds != 0)) {
            trackResults();
        }
        numOfMoves = 0;
    }

    protected void trackResults() {
        //save the current win percentage for each chunk_size
        log.stream.printf("%2.1f\n", 100.0 * totalNumWins / totalNumRounds);
    }

    @Override
    public void onWin(WinEvent event) {
        totalNumWins++;
        reward += goodTerminalReward;
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
        q.save(getDataFile(lutFileName));
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
                new Pair("badInstantReward", badInstantReward),
                new Pair("badTerminalReward", badTerminalReward),
                new Pair("goodInstantReward", goodInstantReward),
                new Pair("goodTerminalReward", goodTerminalReward)
        };
    }

    private void makeDecision() throws IOException {
        if(numOfMoves == 0) //first move
        {
            currentAction = Action.SelectRandomAction();
            return;
        }
        if(currentState.isNotEqual(previousState)) {
            if (totalNumRounds > STOP_RANDOM_ACTION_ROUND){ //stop exploration after initial rounds
                epsilon = 0.0;
            }

            Pair<Action.enumActions, Double> bestActionValue = null;
            //take action
            if(Math.random() > epsilon )
            {
                bestActionValue = this.getBestAction(currentState);
                currentAction = bestActionValue.getKey();
            }else{ //try exploration
                currentAction = Action.SelectRandomAction();
            }

            prePerformingValueFunctions();

            performValueFunction(bestActionValue);

            postPerformingValueFunctions();

            //reset reward
            reward = 0;

        }

        if(currentAction == Action.enumActions.fire && getGunHeat() != 0)
            reward += badInstantReward;

    }

    protected void prePerformingValueFunctions() throws IOException {
    }

    protected void postPerformingValueFunctions() throws IOException {
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
    }
}
