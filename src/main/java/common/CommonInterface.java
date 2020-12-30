package common;

import java.io.File;
import java.io.IOException;

public interface CommonInterface {
    /**
     * @param inputs The input vector. An array of doubles.
     * @return The values returned by th LUT or NN for this input vector. That values can be single or multiple outputs
     */
    double[] outputFor(double[] inputs);

    /**
     * This method will tell the NN or the LUT the output
     * value that should be mapped to the given input vector. I.e.
     * the desired correct output value for an input.
     *
     * @param inputs        The input vector
     * @param expectedOutputs The new value to learn
     * @return The error in the output for that input vector, that is the total of RMS error of the outputs
     */
    double train(double[] inputs, double[] expectedOutputs);

    /**
     * A method to write either a LUT or weights of an neural net to a file.
     *
     * @param argFile of type File.
     */
    void save(File argFile) throws IOException;

    /**
     * Loads the LUT or neural net weights from file. The load must of course
     * have knowledge of how the data was written out by the save method.
     * You should raise an error in the case that an attempt is being
     * made to load data into an LUT or neural net whose structure does not match
     * the data in the file. (e.g. wrong number of hidden neurons).
     *
     * @throws IOException potential io exception
     */
    void load(String argFileName) throws IOException;
}
