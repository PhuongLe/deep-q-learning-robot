package backpropagation;

import common.Activation;

public class SigmoidActivation implements Activation {
    private double argumentA;
    private double argumentB;
    public SigmoidActivation(double argA, double argB){
        this.argumentA = argA;
        this.argumentB = argB;
    }
    @Override
    public double ComputeY(double x) {
        return (argumentB - argumentA) / (1 + Math.pow(Math.E, -x)) + argumentA;
    }

    @Override
    public double ComputeDerivative(double y) {
        // given that f(x) = (b-a)/(1 + e^-x) + a
        // => g(x) = 2*f(x) - 1
        // binary: error * f(x) * (1 - f(x))
        // bipolar: error * 0.5 (1 - g(x) * (1 + g(x))
        return (1/(argumentB - argumentA))*(y - argumentA) * (argumentB - y);    }
}
