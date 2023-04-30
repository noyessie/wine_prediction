package com.njit.edu.cs643.v2;

import com.njit.edu.cs643.v2.helper.ColumnWeightTransformer;
import com.njit.edu.cs643.v2.helper.MyVectorAssember;
import com.njit.edu.cs643.v2.helper.PredictionColumnPrefixer;
import com.njit.edu.cs643.v2.helper.RenameColumnTransformer;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.Params;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

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
        SparkConf conf  = new SparkConf().setAppName("CS 643 Programming Assigment 2")
                .setMaster("local[*]");

        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        System.setOut(new PrintStream(new FileOutputStream(new File("logs/console_"+(new Date().getTime())+".log"))));
        FileWriter writer = new FileWriter(new File("output.txt"));




        Dataset<Row> dataset = spark.read()
                .option("sep" , ";")
                .option("inferSchema" , true)
                .option("quote","\"")
                .option("header" , true)
                .csv(inputFile);

        Dataset<Row> validation = spark.read()
                .option("sep" , ";")
                .option("inferSchema" , true)
                .option("quote","\"")
                .option("header" , true)
                .csv("data/validation.csv");

        //we create the evaluater
        MulticlassClassificationEvaluator f1mesure = new MulticlassClassificationEvaluator()
                .setMetricName("f1")
                .setLabelCol("quality");

        MulticlassClassificationEvaluator accuracy = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy")
                .setLabelCol("quality");

        Pipeline pipeline = new Pipeline();
        List<PipelineStage> stages = new ArrayList<>();

        ParamGridBuilder builder = new ParamGridBuilder(); // for optimization, we will be setting the params to optimize there.

        //we rename the columns to the right name
        RenameColumnTransformer renameColums = new RenameColumnTransformer();
        stages.add(renameColums);


        //assemble features
        MyVectorAssember featureAssembler = new MyVectorAssember();
        stages.add(featureAssembler);

        //adding weight to prediction so we can improove the logistic regression algorithm
        Transformer addWeights = new ColumnWeightTransformer();
        stages.add(addWeights);



        //we can now add the models
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("features")
                .setFamily("multinomial")
                .setMaxIter(10)
                .setRegParam(0.1)
                .setElasticNetParam(0.3)
                .setFamily("multinomial")
                .setWeightCol("weight");



                //params for logistic regression
                builder.addGrid(lr.maxIter() , new int[]{100,250,500,100})
                .addGrid(lr.regParam() , new double[]{0 , 0.01})
                .addGrid(lr.elasticNetParam() , new double[]{0.1,0.3,0.5,0.9})
                .build();
        stages.add(lr);
        stages.add(new PredictionColumnPrefixer("lr_"));


        // Random Forest Classifier
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("quality")
                .setFeaturesCol("features");
        builder.addGrid(rf.minInfoGain() , new double[]{0.00, 0.1 , 0.01})
                .addGrid(rf.maxDepth() , new int[]{15 , 20, 25, 30})
                .addGrid(rf.numTrees() , new int[]{20 , 50 , 100 , 250});
        stages.add(rf);
        stages.add(new PredictionColumnPrefixer("rf_"));

        //adding MLP prediction
        MultilayerPerceptronClassifier mlp = new MultilayerPerceptronClassifier()
                .setLabelCol("quality")
                .setFeaturesCol("features")
                .setLayers(new int[]{11,20,25,20,10})
                .setBlockSize(128);
        builder.addGrid(mlp.maxIter() , new int[]{250,500,1000})
                .addGrid(mlp.blockSize() , new int[]{100});
        stages.add(mlp);
        stages.add(new PredictionColumnPrefixer("mlp_"));

        //Now we merge the 3 result into one pipeline to see if we can get better result


        ParamMap params[] = builder.build();

        pipeline.setStages(stages.stream().toArray(size -> new PipelineStage[size]));
        PipelineModel model = pipeline.fit(dataset);

        Dataset<Row> result = model.transform(dataset);
        result.show(false);

        double lr_p = f1mesure.setPredictionCol("lr_prediction").evaluate(result);
        double rf_p = f1mesure.setPredictionCol("rf_prediction").evaluate(result);
        double mlp_p = f1mesure.setPredictionCol("mlp_prediction").evaluate(result);

        writer.write("F1-measure for Logistic Regression on the training set : "+lr_p+" \n");
        writer.write("F1-measure for Random Forest on the training set : "+rf_p+" \n");
        writer.write("F1-measure for MultiLayer classifier on the training set: "+mlp_p+" \n");

        System.out.println(lr_p + " --- " +rf_p +" --- "+ mlp_p);




    }
}
