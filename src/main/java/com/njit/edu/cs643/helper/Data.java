package com.njit.edu.cs643.helper;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;

import static org.apache.spark.sql.functions.col;

public class Data {

    private Data(){};

    public static Dataset<Row> getData(SparkSession spark , String path){
        Dataset<Row> dataset = spark.read()
                .option("sep" , ";")
                .option("inferSchema" , true)
                .option("quote","\"")
                .option("header" , true)
                .csv(path)
                ;

        // we map the columns name
        dataset = dataset.select(generateNewColumnName(dataset.columns()));

        // convert in a format usable by the model
        VectorAssembler featureAssembler = new VectorAssembler()
                .setInputCols(
                        Arrays.stream(dataset.columns()).filter(col -> !col.equalsIgnoreCase("quality")) // we remove the column quality
                                .toArray(size -> new String[size])
                ).setOutputCol("features");

        dataset = featureAssembler.transform(dataset).withColumnRenamed("quality", "label");

        return dataset;
    }

    public static Dataset<Row> getTrainingData(SparkSession spark ){
        return getData(spark , "data/training.csv");
    }

    public static Dataset<Row> getValidationData(SparkSession spark ){
        return getData(spark , "data/validation.csv");
    }

    public static Column[] generateNewColumnName(String[] columns){
        return Arrays.asList(columns).stream()
                .map(x -> col(x).as(x.replaceAll("\"","")))
                .toArray(size -> new Column[size]);
    }

}
