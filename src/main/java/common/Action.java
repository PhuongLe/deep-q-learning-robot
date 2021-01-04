package common;

import java.util.Random;

public class Action {
    public enum enumActions {attack, avoid, runaway, fire};
    public static final int NUM_ACTIONS = enumActions.values().length;

    public static enumActions SelectRandomAction() {
        Random rand = new Random();
        int r = rand.nextInt(enumActions.values().length);
        return enumActions.values()[r];
    }
}
