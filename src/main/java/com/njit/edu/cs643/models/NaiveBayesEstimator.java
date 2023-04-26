package com.njit.edu.cs643.models;

import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;

public class NaiveBayesEstimator implements ModelEstimator {

    @Override
    public CrossValidator getEstimator() {

        CrossValidator validator = new CrossValidator();

        NaiveBayes by = new NaiveBayes()
                ;

        by.explainParams();

        ParamMap[] paramGrid = new ParamGridBuilder()
                .build();


        return validator.setEstimator(by)
                .setEvaluator(new MulticlassClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5)
                .setParallelism(5);
    }
}
