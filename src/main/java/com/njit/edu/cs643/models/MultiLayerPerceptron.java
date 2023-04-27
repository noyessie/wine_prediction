package com.njit.edu.cs643.models;

import com.njit.edu.cs643.helper.DataTransform;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
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

    private static final int[] maxIters = new int[]{1000};
    private static final int[] blockSize = new int[]{150};// brdy mofrl 150
    private static final long[] seed = new long[]{100};


    @Override
    public CrossValidator getEstimator(int folks) {

        //int[] layer = new int[]{11,50,100,50,10};
        int[] layer = new int[]{11,20,25,20,10};


        CrossValidator validator = new CrossValidator();

        MultilayerPerceptronClassifier p = new MultilayerPerceptronClassifier()
                .setLayers(layer)
                ;
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{DataTransform.getScalar() , p});


        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(p.maxIter() , maxIters)
                .addGrid(p.blockSize() ,blockSize)
                .addGrid(p.seed() , seed)

                .build();


        return validator.setEstimator(p)
                .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("f1"))
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(folks)
                .setParallelism(30);
    }
}
