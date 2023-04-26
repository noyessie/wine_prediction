package com.njit.edu.cs643.data;

import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import javax.xml.crypto.Data;
import java.util.Arrays;

import static org.apache.spark.sql.functions.col;

public final class DataExtractor {
    private DataExtractor(){

    }

    public static Dataset<Row> getData(SparkSession spark, String path , boolean clean , boolean standardize){
        Dataset<Row> dataset = spark.read()
                .option("sep" , ";")
                .option("inferSchema" , true)
                .option("quote","\"")
                .option("header" , true)
                .csv(path)
                ;

        // we map the columns name
        dataset = dataset.select(
                Arrays.asList(dataset.columns()).stream()
                        .map(x -> col(x).as(x.replaceAll("\"","")))
                        .toArray(size -> new Column[size])
        );

        // convert in a format usable by the model
        VectorAssembler featureAssembler = new VectorAssembler()
                .setInputCols(
                        Arrays.stream(dataset.columns()).filter(col -> !col.equalsIgnoreCase("quality")) // we remove the column quality
                                .toArray(size -> new String[size])
                ).setOutputCol("features");

        dataset = featureAssembler.transform(dataset).withColumnRenamed("quality", "label");

        if(standardize){
            RobustScaler scaler = new RobustScaler()
                    .setInputCol("features")
                    .setOutputCol("stdfeatures")
                    .setWithCentering(true)
                    .setWithScaling(true);

            RobustScalerModel scalerModel = scaler.fit(dataset);

            StandardScaler sscaler = new StandardScaler()
                    .setInputCol("features")
                    .setOutputCol("ssfeatures")
                    .setWithMean(true)
                    .setWithStd(true);

            StandardScalerModel sscalerModel = sscaler.fit(dataset);

            dataset = sscalerModel.transform(scalerModel.transform(dataset))
                    .drop("features")
                    .withColumnRenamed("ssfeatures","features");
        }
        if(clean){
            return dataset.select("features" , "label");
        }else{
            return dataset;
        }

    }


    public static Dataset<Row> getTrainingData(SparkSession spark, boolean clean , boolean std){
        return getData(spark , "data/training.csv" , clean , std);

    }

    public static Dataset<Row> getTrainingData(SparkSession spark){
        return getTrainingData(spark , false , false);
    }

    public static Dataset<Row> getValidationData(SparkSession spark, boolean clean , boolean std){
        return getData(spark , "data/validation.csv" , clean , std);

    }

    public static Dataset<Row> getValidatioData(SparkSession spark){
        return getTrainingData(spark , false , false) ;
    }

}
