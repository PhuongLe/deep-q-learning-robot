package common;

public class Experience {
    public State previousState;
    public Action.enumActions previousAction;
    public double currentReward;
    public State currentState;
    public Experience(State s1, Action.enumActions a1, double r, State s2){
        this.previousAction = a1;
        this.previousState = s1;
        this.currentReward = r;
        this.currentState = s2;
    }
}
