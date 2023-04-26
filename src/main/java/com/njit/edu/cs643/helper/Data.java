package com.njit.edu.cs643.helper;

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
        return dataset;
    }

    public static Dataset<Row> getTrainingData(SparkSession spark ){
        return getData(spark , "data/training.csv");
    }

    public static Dataset<Row> getValidationData(SparkSession spark ){
        return getData(spark , "data/validation.csv");
    }

    public static Column[] renameColumns(String[] columns){
        return Arrays.asList(columns).stream()
                .map(x -> col(x).as(x.replaceAll("\"","")))
                .toArray(size -> new Column[size]);
    }

}
