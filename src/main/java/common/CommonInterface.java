package common;

import java.io.File;
import java.io.IOException;

public interface CommonInterface {
    /**
     * This function is used for getting the corresponding value of a state-action space vector from a LUT or NN.
     * @param stateActionSpaceVector The state-action space vector. An array of doubles.
     * @return The value returned by th LUT or NN for this state-action space vector.
     *         It would be the corresponding Q-Value of the state-action space vector
     */
    double outputFor(double[] stateActionSpaceVector);

    /**
     * This method will tell the NN or the LUT the target value that should be mapped to the given state-action vector. I.e.
     * the desired correct output value for a state-action space.
     *
     * @param stateActionSpaceVector The state-action space vector. An array of doubles.
     * @param target The new value to learn. It would be properly the corresponding new Q-Value of the state-action space vector.
     * @return The error in the output for that state-action space vector
     */
    double train(double[] stateActionSpaceVector, double target);

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
