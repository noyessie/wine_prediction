package com.njit.edu.cs643.helper;

import org.apache.spark.ml.feature.StandardScaler;

public class DataTransform {

    private DataTransform(){}
;

    public static StandardScaler getScalar(){
        return new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaled_features");
    }
}
