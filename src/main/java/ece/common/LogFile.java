package ece.common;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import javafx.util.Pair;
import robocode.RobocodeFileOutputStream;

/**
 * @author sarbjit
 * {@docRoot}
 * @version 0.1
 * This class implements a file logging mechanism. It is meant to enable
 * diagnostic data to be written from a robocode tank to a file.
 */
public class LogFile {
    /**
     * Private members of this class
     */
    public PrintStream stream;

    /**
     * Constructor.
     * @param argFile The file created by Robocode into which to write data
     *
     */
    public LogFile ( File argFile ) {
        try {
            stream = new PrintStream( new RobocodeFileOutputStream( argFile ));
            System.out.println( "--+ Log file created." );
        } catch (IOException e) {
            System.out.println( "*** IO exception during file creation attempt.");
        }
    }
    public void printHyperParameters(Pair<String, Double>[] parameters){
        for (Pair<String, Double> parameter : parameters) {
            stream.printf("%s:   %2.2f\n", parameter.getKey(), parameter.getValue());
        }
    }
    public void closeStream(){
        if(stream!=null){
            stream.close();
        }
    }
} // End of public class LogFile
