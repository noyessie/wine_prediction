package com.njit.edu.cs643.models;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;

public interface ModelEstimator {

    public CrossValidator getEstimator(int folks);

}
