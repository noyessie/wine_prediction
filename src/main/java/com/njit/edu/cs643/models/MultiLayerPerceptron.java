package com.njit.edu.cs643.models;

import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MultiLayerPerceptron implements ModelEstimator {

    private static final int[] maxIters = new int[]{100};
    private static final int[] blockSize = new int[]{150};
    private static final long[] seed = new long[]{100};


    @Override
    public CrossValidator getEstimator() {

        int[] layer = new int[]{11,25,20,10};

        CrossValidator validator = new CrossValidator();

        MultilayerPerceptronClassifier p = new MultilayerPerceptronClassifier()
                .setLayers(layer)
                ;


        p.explainParams();

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(p.maxIter() , maxIters)
                .addGrid(p.blockSize() ,blockSize)
                .addGrid(p.seed() , seed)

                .build();


        return validator.setEstimator(p)
                .setEvaluator(new MulticlassClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5)
                .setParallelism(5);
    }
}
