package ece.backpropagation;

import ece.common.Activation;

public class TanhActivation implements Activation {
    @Override
    public double ComputeY(double x) {
        return Math.tanh(x);
    }

    @Override
    public double ComputeDerivative(double y) {
        return 1 - Math.pow(Math.tanh(y), 2);
    }
}
