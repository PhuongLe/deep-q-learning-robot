package ece.common;

import ece.backpropagation.StateActionNeuralNet;

public class State {
    public enum enumEnergy{low, medium, high}
    public enum enumDistance{veryClose, near, far}
    public enum enumGunHeat{low, high}

    public static final int NUM_ENERGY = enumEnergy.values().length;
    public static final int NUM_DISTANCE = enumDistance.values().length;
    public static final int NUM_GUN_HEAT = enumGunHeat.values().length;

    private static final int distanceStep = 150;

    public enumEnergy energy;
    public enumDistance distance;
    public enumGunHeat gunHeat;
    public double bearing = 0;

    public State()
    {
        initialize();
    }

    void initialize()
    {
        energy=enumEnergy.high;
        distance=enumDistance.near;
        gunHeat = enumGunHeat.low;
    }

    public void update(enumEnergy energy,
                enumDistance distance,
                enumGunHeat gunHeat,
                double bearing)
    {
        //Update Variables
        this.energy = energy;
        this.distance = distance;
        this.gunHeat = gunHeat;
        this.bearing = bearing;
    }

    public void update(double energy,
            double distance,
            double gunHeat,
            double bearing)
    {
        //Update Variables
        this.energy = enumEnergyOf(energy);
        this.distance = enumDistanceOf(distance);
        this.gunHeat = enumGunHeatOf(gunHeat);
        this.bearing = bearing;
    }

    public double[] StateActionValue(int action){
        return new double[]{this.energy.ordinal(), this.distance.ordinal(), this.gunHeat.ordinal(), action};
    }

    public double[] StateActionInputVector(int action){
        return StateActionNeuralNet.MapStateActionToInputVector(this.energy, this.distance, this.gunHeat, Action.enumActions.values()[action]);
    }

    public boolean isNotEqual(State previousState) {
        return (!this.energy.equals(previousState.energy)
                || !this.distance.equals(previousState.distance)
                || !this.gunHeat.equals(previousState.gunHeat));
    }


    private enumGunHeat enumGunHeatOf(double gunHeat) {
        if(gunHeat > 0)
            return enumGunHeat.high;

        return enumGunHeat.low;
    }

    public enumEnergy enumEnergyOf(double energy) {
        if (energy < 15)
            return enumEnergy.low;
        else if (energy>=15 && energy < 50)
            return enumEnergy.medium;

        return enumEnergy.high;
    }

    public enumDistance enumDistanceOf(double distance) {
        int newDistance = Math.abs((int) distance / distanceStep);
        if (newDistance < NUM_DISTANCE)
            return enumDistance.values()[newDistance];
        else
            return enumDistance.values()[NUM_DISTANCE - 1];
    }
}
