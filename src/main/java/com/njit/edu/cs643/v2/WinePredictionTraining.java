package com.njit.edu.cs643.v2;

import com.njit.edu.cs643.v2.helper.*;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
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

        spark.sparkContext().hadoopConfiguration().set("fs.s3n.awsAccessKeyId", "awsAccessKeyId value");
        spark.sparkContext().hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", "awsSecretAccessKey value");
        spark.sparkContext().hadoopConfiguration().set("fs.s3n.endpoint", "s3.amazonaws.com");
        spark.sparkContext().hadoopConfiguration().set("spark.hadoop.fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem");


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


        Pipeline pipeline = new PipelineBuilder().getPipeline();
        PipelineModel model = pipeline.fit(dataset);
        Dataset<Row> result = model.transform(dataset);
        result.show(false);

        /*double lr_p = f1mesure.setPredictionCol("lr_prediction").evaluate(result);
        double rf_p = f1mesure.setPredictionCol("rf_prediction").evaluate(result);
        double mlp_p = f1mesure.setPredictionCol("mlp_prediction").evaluate(result);

        writer.write("F1-measure for Logistic Regression on the training set : "+lr_p+" \n");
        writer.write("F1-measure for Random Forest on the training set : "+rf_p+" \n");
        writer.write("F1-measure for MultiLayer classifier on the training set: "+mlp_p+" \n");
        writer.close();
        System.out.println(lr_p + " --- " +rf_p +" --- "+ mlp_p);*/

        model.write().overwrite().save(outputFolder+"/models");
        model.write().overwrite().save(outputFolder+"/models_"+Utils.getTimestamp());

    }
}
