package com.njit.edu.cs643;

import com.njit.edu.cs643.helper.Data;
import com.njit.edu.cs643.helper.DataTransform;
import com.njit.edu.cs643.models.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.util.MLWritable;
import org.apache.spark.sql.*;
import scala.Tuple2;

import java.io.*;
import java.util.Arrays;
import java.util.Date;

import static org.apache.spark.sql.functions.*;


public class WinePrediction {

    public static void main(String[] args) throws IOException {

        // configure spark
        SparkConf conf  = new SparkConf().setAppName("CS 643 Programming Assigment 2")
                .setMaster("local[*]");

        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        Dataset<Row> dataset = Data.getTrainingData(spark);
        dataset = DataTransform.getScalar().fit(dataset).transform(dataset);

        final long count = dataset.count();



        System.out.println("count all "+ count);
        Dataset<Row> weights = dataset.groupBy("label")
                .agg(count("label").as("total"))
                .selectExpr("*" , "1- (total / "+count+") as weight")
                .select("label","weight");

        dataset = dataset.join(weights , "label");

        Dataset<Row> validation = Data.getValidationData(spark);
        validation = DataTransform.getScalar().fit(validation).transform(validation);


        MulticlassClassificationEvaluator f1mesure = new MulticlassClassificationEvaluator()
                .setMetricName("f1")
                .setLabelCol("label");

        MulticlassClassificationEvaluator accuracy = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy")
                .setLabelCol("label");

        final ModelEstimator[] estimators = new ModelEstimator[]{
                new MultinomialLogisticRegression(),
                new RandomForests(),
                new MultiLayerPerceptron()
        };

        final Model[] bestModels = new Model[estimators.length];

        final String basePath = "models/";

        String[] paths = new String[]{
                "models/lr.model",
                "models/rf.model",
                "models/mlp.model"
        };

        for(int i = 0 ; i< estimators.length ; i++){
            double bestF1 = 0;
            Model model =null;
            for(int j = 3 ; j<10 ; j++){
                model = estimators[i].getEstimator(j).fit(dataset).bestModel();
                double f1 = f1mesure.evaluate(model.transform(validation));
                if(f1>bestF1)
                {
                    bestModels[i] = model;
                    bestF1 = f1;
                }
            }

            System.out.println("Done with model " + i + "with mesure "+bestF1+" explanation : \n"+ model.explainParams() + "\n\n\n");

            ((MLWritable)model).save(paths[i]);
        }

        

        //average the modesl
        Dataset<Row> predictions = bestModels[0].transform(validation).select(column("label") , column("prediction").as("lgr"))
                .withColumn("id",monotonically_increasing_id())
                .join(
                        bestModels[1].transform(validation).select(column("prediction").as("forest"))
                                .withColumn("id",monotonically_increasing_id()),
                        "id","outer"
                ).join(
                        bestModels[2].transform(validation).select(column("prediction").as("nlp"))
                                .withColumn("id",monotonically_increasing_id()),
                        "id","outer"
                ).selectExpr("*" , "if(lgr==forest , lgr , 0) as one" , "if(lgr==nlp , lgr , 0) as two" , "if(nlp==forest , nlp , 0) as three")
                .selectExpr("*" , "array_max(array(one , two , three)) as post_prediction")
                .selectExpr("*", "if(post_prediction == 0 , lgr , post_prediction) as prediction");

        predictions.printSchema();
        predictions.show(false);

        System.out.println("Mesure " + f1mesure.evaluate(predictions));
        System.out.println("Accuracty " + accuracy.evaluate(predictions));


    }
}
