package com.njit.edu.cs643.models;

import com.njit.edu.cs643.helper.DataTransform;
import com.njit.edu.cs643.helper.RenameColumnTransformer;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;

import java.util.Arrays;

public class NaiveBayesEstimator implements ModelEstimator {

    @Override
    public CrossValidator getEstimator(int folks) {

        CrossValidator validator = new CrossValidator();

        NaiveBayes by = new NaiveBayes()
                .setSmoothing(0.1);

                //.setFeaturesCol("scaled_features");

        System.out.println(by.explainParams());

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{DataTransform.getScalar() , by});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .build();


        return validator.setEstimator(by)
                .setEvaluator(new MulticlassClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(folks)
                .setParallelism(folks);
    }
}
