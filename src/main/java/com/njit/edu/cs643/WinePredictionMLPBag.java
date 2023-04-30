package com.njit.edu.cs643;

import com.njit.edu.cs643.helper.Data;
import com.njit.edu.cs643.helper.DataTransform;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.ColumnPruner;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorAttributeRewriter;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.collection.immutable.Set;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Collections;
import java.util.Date;

import static org.apache.spark.sql.functions.column;

public class WinePredictionMLPBag {
    public static void main(String[] args) throws FileNotFoundException {

        SparkConf conf  = new SparkConf().setAppName("CS 643 Programming Assigment 2")
                .setMaster("local[*]");

        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        System.setOut(new PrintStream(new FileOutputStream(new File("logs/console_"+(new Date().getTime())+".log"))));


        Dataset<Row> dataset = Data.getTrainingData(spark);
        dataset = DataTransform.getScalar().fit(dataset).transform(dataset);

        Dataset<Row> validation = Data.getValidationData(spark);
        validation = DataTransform.getScalar().fit(validation).transform(validation);
        final String basePath = "models/";

        RandomForestClassificationModel forest = RandomForestClassificationModel.load(basePath+"rf.model");
        LogisticRegressionModel lrm = LogisticRegressionModel.load(basePath+"lr.model");
        MultilayerPerceptronClassificationModel mlpm = MultilayerPerceptronClassificationModel.load(basePath+"mlp.model");

        MulticlassClassificationEvaluator f1mesure = new MulticlassClassificationEvaluator()
                .setMetricName("f1")
                .setLabelCol("label");

        MulticlassClassificationEvaluator accuracy = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy")
                .setLabelCol("label");

        Model bestModels[] = new Model[]{
                lrm , forest, mlpm
        };
        for(int i =0 ; i< 3 ; i++){
            Model m = bestModels[i];
            System.out.println("Best model for the classifier " + i + " with a precision of " + f1mesure.evaluate(m.transform(validation)));
            System.out.println(m.explainParams());
        }

        MultilayerPerceptronClassifier mlpc = new MultilayerPerceptronClassifier()
                .setLayers(new int[]{3 , 20 ,50, 30 , 10})
                .setFeaturesCol("learned_features")
                .setMaxIter(1000)
                .setStepSize(.01);
        VectorAssembler assembler = new VectorAssembler()
//                .setInputCols(new String[]{"lrm","lrm-prob" , "forest" , "forest-prob","mlpm" , "nlpm-prob"})
                .setInputCols(new String[]{"lrm", "forest" , "mlpm" , })
                .setOutputCol("learned_features");


        Pipeline pipeline = new Pipeline()
                .setStages(
                        new PipelineStage[]{
                                lrm.setPredictionCol("lrm")
                                        .setProbabilityCol("lrm-prob")
                                        .setRawPredictionCol("lmr-raw"),
                                forest.setPredictionCol("forest")
                                        .setProbabilityCol("forest-prob")
                                        .setRawPredictionCol("forest-raw"),

                                mlpm.setPredictionCol("mlpm").setRawPredictionCol("nlpm-raw")
                                        .setProbabilityCol("nlpm-prob")
                                        .setRawPredictionCol("nlpm-raw")
                                        ,
                                assembler,
                                mlpc
                        }
                );

        Dataset<Row> predictions = pipeline.fit(dataset).transform(validation);

        predictions.show(false);

        System.out.println("f1 mesure "+f1mesure.evaluate(predictions));
        System.out.println("accuracy "+accuracy.evaluate(predictions));
    }
}
