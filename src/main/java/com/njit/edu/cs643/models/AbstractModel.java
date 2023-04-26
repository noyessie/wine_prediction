package com.njit.edu.cs643.models;

import org.apache.spark.ml.param.ParamMap;

public abstract class AbstractModel implements ModelEstimator {

    ParamMap[] paramGrid;

    public AbstractModel(){
        this.initParamsGrid();
    }

    public abstract void initParamsGrid();
}
