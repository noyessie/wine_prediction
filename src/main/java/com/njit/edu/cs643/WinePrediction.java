package com.njit.edu.cs643;

import com.njit.edu.cs643.data.DataExtractor;
import com.njit.edu.cs643.models.ModelEstimator;
import com.njit.edu.cs643.models.MultiLayerPerceptron;
import com.njit.edu.cs643.models.MultinomialLogisticRegression;
import com.njit.edu.cs643.models.RandomForests;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.*;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;

import static org.apache.spark.sql.functions.col;


public class WinePrediction {

    public static void main(String[] args) throws FileNotFoundException {

        //redirect the standard output to file, good for testing
//        FileOutputStream fos = new FileOutputStream(new File("console.log"));
//        PrintStream ps = new PrintStream(fos);
//
//        System.setOut(ps);

        // configure spark
        SparkConf conf  = new SparkConf().setAppName("CS 643 Programming Assigment 2")
                .setMaster("local[*]");

        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> dataset = DataExtractor.getTrainingData(spark , true , true);
        Dataset<Row> validation = DataExtractor.getValidationData(spark , false , true);



        //ModelEstimator ml = new MultinomialLogisticRegression();
        //ModelEstimator ml = new RandomForests();
        ModelEstimator ml = new MultiLayerPerceptron();

        CrossValidatorModel model = ml.getEstimator().fit(dataset);

        Model bestModel = model.bestModel();



        System.out.println("Model wining params" + bestModel.explainParams());
//        System.out.println("validation");
//        validation.show();

        Dataset<Row> predictions = bestModel.transform(validation);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy")
                .setLabelCol("label");

        double accuracy = evaluator.evaluate(predictions);

        System.out.println("Accurracy " + accuracy);


    }
}
