package com.njit.edu.cs643.models;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;

public class RandomForests implements ModelEstimator {

    private static final double[] minInfoGain = new double[]{0.0 , 0.01,0.001};

    private static final int[] maxDepth = new int[]{5 , 7, 9 , 12, 15 , 20, 25, 30};
    @Override
    public CrossValidator getEstimator() {

        CrossValidator validator = new CrossValidator();

        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                ;



        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(dt.minInfoGain() , minInfoGain)
                .addGrid(dt.maxDepth() , maxDepth)
                .build();

        System.out.println(dt.getMinInfoGain());

        return validator.setEstimator(dt)
                .setEvaluator(new MulticlassClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5)
                .setParallelism(5);
    }
}
