package com.njit.edu.cs643.models;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;

public class MultinomialLogisticRegression implements ModelEstimator {

    private static final int[] maxIters = new int[]{100 , 200 , 500 , 1000};
    private static final double[] regparams = new double[]{0.01,0.1,0.2,0.3,0.4};
    private static final double[] elasticNetParams = new double[]{0.1,0.3,0.5,0.9};
    @Override
    public CrossValidator getEstimator() {

        CrossValidator validator = new CrossValidator();

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.1)
                .setElasticNetParam(0.3)
                .setFamily("multinomial");

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(lr.maxIter() , maxIters)
                .addGrid(lr.regParam() , regparams)
                .addGrid(lr.elasticNetParam() , elasticNetParams)
                .build();

        return validator.setEstimator(lr)
                .setEvaluator(new MulticlassClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5)
                .setParallelism(5);
    }
}
