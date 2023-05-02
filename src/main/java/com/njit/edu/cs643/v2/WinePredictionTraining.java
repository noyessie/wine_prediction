package com.njit.edu.cs643.v2;

import com.njit.edu.cs643.v2.helper.*;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.*;
import java.util.Date;

public class WinePredictionTraining {
    public static void main(String[] args) throws IOException {


        if(args.length<2){
            System.err.println("Invalid number of arguments. Specify the <input file> and the <ouput file path> " + args.length );
            System.exit(0);
        }

        //parameters
        String inputFile = args[0];
        String outputFolder = args[1];



        // configure spark
        SparkConf conf  = new SparkConf().setAppName("CS 643 Programming Assigment 2");
                //.setMaster("local[*]");

        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();


        //System.setOut(new PrintStream(new FileOutputStream(new File("logs/console_"+(new Date().getTime())+".log"))));
        //FileWriter writer = new FileWriter(new File("output.txt"));

        Dataset<Row> dataset = spark.read()
                .option("sep" , ";")
                .option("inferSchema" , true)
                .option("quote","\"")
                .option("header" , true)
                .csv(inputFile);

        //we create the evaluater
        MulticlassClassificationEvaluator f1mesure = new MulticlassClassificationEvaluator()
                .setMetricName("f1")
                .setLabelCol("quality");

        MulticlassClassificationEvaluator accuracy = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy")
                .setLabelCol("quality");


        PipelineBuilder b = new PipelineBuilder();
        Pipeline pipeline = b.getPipeline();

        CrossValidator validator = new CrossValidator();

        validator.setEstimator(pipeline)
                .setEvaluator(f1mesure)
                .setEstimatorParamMaps(b.getParams())
                .setNumFolds(4) // some class have 9 items. so too much folk is not good.
                .setParallelism(250);

        System.out.println("params map validator size "+ validator.getEstimatorParamMaps().length);

        CrossValidatorModel model = validator.fit(dataset);
//
//        PipelineModel model = pipeline.fit(dataset);
        Dataset<Row> result = model.transform(dataset);
        result.show(false);


        PipelineModel bestM = (PipelineModel) model.bestModel();

        bestM.write().overwrite().save(outputFolder+"/models");
        bestM.write().overwrite().save(outputFolder+"/models_"+Utils.getTimestamp());

    }
}
