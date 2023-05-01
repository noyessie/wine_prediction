package com.njit.edu.cs643;

import com.njit.edu.cs643.v2.helper.PredictionColumnPrefixer;
import com.njit.edu.cs643.v2.helper.RenameColumnTransformer;
import com.njit.edu.cs643.v2.helper.RenameColumnTransformer2;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class Test {

    public static void main(String args[]) throws IOException {

        SparkConf conf  = new SparkConf().setAppName("CS 643 Programming Assigment 2")
                .setMaster("local[*]");

        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        Transformer trf = new PredictionColumnPrefixer().setPrefixCol("hubert");
        //Transformer trf = new RenameColumnTransformer();

        Pipeline p = new Pipeline();
        p.setStages(new PipelineStage[]{trf});

        p.write().overwrite().save("mytransformer");
        System.out.println("Let try to load ");

        p = Pipeline.load("mytransformer");
        System.out.println(p.explainParams());
        //Pipeline p = Pipeline.load("mytransformer");
        for(PipelineStage stage: p.getStages()){
            System.out.println("The current stage is "+stage.logName());
            System.out.println(" Load :::: "+stage.explainParams());
        }
    }
}
