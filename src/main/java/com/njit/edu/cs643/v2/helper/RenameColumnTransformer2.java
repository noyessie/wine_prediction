package com.njit.edu.cs643.v2.helper;

import java.io.IOException;
import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class RenameColumnTransformer2 extends Transformer implements MLWritable {

    private String uid;

    public RenameColumnTransformer2(String uid) {
        this.uid = uid;
    }

    public RenameColumnTransformer2() {
        this.uid = "Cleanser" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> sentences) {
        return sentences.select("*");
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema;
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<RenameColumnTransformer2> read() {
        return new DefaultParamsReader<>();
    }

}