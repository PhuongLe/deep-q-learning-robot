package ece.backpropagation;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Scanner;

public class Prototyping {

    public static void main(String[] args) throws IOException {

        NN_test();

    }

    public static void NN_test() throws IOException {
        int numInputs = 8;
        int numHiddenLayerNeurons = 10;
        int numOutput = 1;
        float learningRate = 0.01f;
        float momentumValue = 0.9f;
        double sigmoidLB = -1.0;
        double sigmoidUB = 1.0;
        NeuralNetED nn = new NeuralNetED(numOutput, numInputs, numHiddenLayerNeurons, learningRate, momentumValue, sigmoidLB, sigmoidUB);
    }


    private static void read_file() throws Exception{

        Scanner input = new Scanner(new File("rmse_error.txt"));

        String arr_str = "";
        while (input.hasNext()){
            arr_str = arr_str+""+input.next();
        }

        String[] modified_nbr = arr_str.split(",");
        Double[] dbl_arr = new Double[modified_nbr.length];
        for(int i = 0; i< modified_nbr.length; i++){
            String tmp_str = modified_nbr[i];
            tmp_str = tmp_str.replaceAll("[^0-9\\.]", "");
            dbl_arr[i] = Double.parseDouble(tmp_str);
        }

        System.out.println(Arrays.toString(dbl_arr));

    }

    private static void  write_to_file(String file_path, String content) throws Exception
    {
        PrintStream out = new PrintStream(new FileOutputStream(file_path));
        out.print(content);
    }

}
