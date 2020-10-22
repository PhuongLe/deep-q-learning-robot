package ece.backpropagation;

import ece.common.Activation;

public class ReLuActivation implements Activation {
    @Override
    public double ComputeY(double x) {
        return x > 0 ? x : 0;
    }

    @Override
    public double ComputeDerivative(double y) {
        return y > 0 ? y : 0;
    }
}
