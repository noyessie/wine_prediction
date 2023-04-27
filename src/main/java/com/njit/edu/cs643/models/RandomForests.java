package com.njit.edu.cs643.models;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;

public class RandomForests implements ModelEstimator {

    private static final double[] minInfoGain = new double[]{0.00, 0.1 , 0.01}; // optimized value: 0.01

    private static final int[] maxDepth = new int[]{15 , 20, 25, 30};//optimized value obtainer: 15,20

    final int[] numTrees = new int[]{20 , 50 , 100 , 250};
    @Override
    public CrossValidator getEstimator(int folks) {

        CrossValidator validator = new CrossValidator();

        RandomForestClassifier dt = new RandomForestClassifier()
                ;

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(dt.minInfoGain() , minInfoGain)
                .addGrid(dt.maxDepth() , maxDepth)
                .addGrid(dt.numTrees() , numTrees)
                .build();

        System.out.println(dt.getMinInfoGain());

        return validator.setEstimator(dt)
                .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("f1"))
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(folks)
                .setParallelism(30);
    }
}
