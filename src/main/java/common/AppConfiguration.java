package common;

import java.text.SimpleDateFormat;
import java.util.Date;

public class AppConfiguration {

    static String BaseFolder = "d:\\Github\\robocode\\report\\";
    static String DataBaseFolder = "d:\\Github\\robocode\\data\\";
    public static String FilePrefix = BaseFolder + new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss") .format(new Date());

    /**
     * File name of the file storing referenced look up table which was saved after the robot being trained by Q-Learning alg.
     */
    public static String ReferencedLutTableFileName = DataBaseFolder+ "lut.log";

    /**
     * File name of the file storing pre-trained neural network's weights after the network being offline trained by runner.
     */
    public static String PretrainedNetworkFileName = DataBaseFolder+ "nn_weights.log";

    //public static String LogFileName = FilePrefix + "-neural-net.log";

    /**
     * File name of the neural network runner's log file
     */
    public static String RunnerReportFileName = FilePrefix + "-runner.log";

    /**
     * File name of the file that saves all neural networks' weights generated by neural network runner
     */
    public static String NetworkWeightsFileName = FilePrefix + "-nn_weights.log";

    /**
     * File name of the robot's log file
     */
    public static String RoboLogFileName = FilePrefix + "-robocode.log";

    /**
     * File name of the robot's lookup table that is saved after each battle
     */
    public static String RoboLutFileName = FilePrefix + "-robocode-lut.log";
}