package com.njit.edu.cs643.v2;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import java.io.*;
import java.util.Arrays;
import java.util.Date;
import java.util.List;


public class WinePredictionValidation {
    public static void main(String[] args) throws IOException {


        if(args.length<3){
            System.err.println("Invalid number of arguments. Specify the <input file> and the <ouput file path> " + args.length );
            System.exit(0);
        }

        //parameters
        String modelFolder = args[0];
        String inputFile = args[1];
        String outputFolder = args[2];

        // configure spark
        SparkConf conf  = new SparkConf().setAppName("CS 643 Programming Assigment 2");
                //.setMaster("local[*]");

        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        //System.setOut(new PrintStream(new FileOutputStream(new File("logs/console_validation_"+(new Date().getTime())+".log"))));
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

        PipelineModel model = PipelineModel.load(modelFolder);

        Dataset<Row> result = model.transform(dataset);
        result.show(false);

        double lr_p = f1mesure.setPredictionCol("lr_prediction").evaluate(result);
        double rf_p = f1mesure.setPredictionCol("rf_prediction").evaluate(result);
        double mlp_p = f1mesure.setPredictionCol("mlp_prediction").evaluate(result);
        double total = f1mesure.setPredictionCol("prediction").evaluate(result);

//        writer.write("F1-measure for Logistic Regression on the training set : "+lr_p+" \n");
//        writer.write("F1-measure for Random Forest on the training set : "+rf_p+" \n");
//        writer.write("F1-measure for MultiLayer classifier on the training set: "+mlp_p+" \n");
//        writer.close();

        // writing the output file to spark output folder
        /*List<Row> f1 = Arrays.asList(
                RowFactory.create("F1-meassure for logistic regression validation set", lr_p),
                RowFactory.create("F1-meassure for random forest  on the validation set", rf_p),
                RowFactory.create("F1-meassure for Multilayer perceptron on the validation set", mlp_p),
                RowFactory.create("F1-meassure for using logistic regression to merge the models", total)
        );*/

        List<Tuple2<String , Double>> f1= Arrays.asList(
          new Tuple2<>("F1-meassure for logistic regression validation set", lr_p),
          new Tuple2<>("F1-meassure for random forest  on the validation set", rf_p),
                new Tuple2<>("F1-meassure for Multilayer perceptron on the validation set", mlp_p),
                new Tuple2<>("F1-meassure for using logistic regression to merge the models", total)
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("measure", DataTypes.StringType, false, Metadata.empty()),
                new StructField("value", DataTypes.DoubleType, false, Metadata.empty())
        });


        Encoder<Tuple2<String, Double>> encoder = Encoders.tuple(Encoders.STRING(), Encoders.DOUBLE());
        Dataset<Tuple2<String , Double>> f1MeasureDataset= spark.createDataset(f1 , encoder);
        result = result.drop("features" , "lr_rawPrediction" ,
                "rf_rawPrediction" , "mlp_rawPrediction" ,
                "lr_probability" , "rf_probability","mlp_probability",
                "rawPrediction" , "probability","composite_feature");

        result.write().mode("overwrite").format("csv").option("header" , "true").save(outputFolder+"/predictions");
        f1MeasureDataset.write().mode("overwrite").format("csv").option("header" , "true").save(outputFolder+"/f1measure");
        System.out.println(lr_p + " --- " +rf_p +" --- "+ mlp_p);





    }
}
