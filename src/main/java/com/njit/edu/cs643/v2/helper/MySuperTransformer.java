package com.njit.edu.cs643.v2.helper;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.Params;
import org.apache.spark.ml.util.*;

import java.io.IOException;

public abstract class MySuperTransformer extends Transformer   implements MLWritable {


    static void init(final DefaultParamsWritable $this) {
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<RenameColumnTransformer> read() {
        return new DefaultParamsReader<>();
    }
}
