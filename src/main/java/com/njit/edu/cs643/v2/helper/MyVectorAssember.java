package com.njit.edu.cs643.v2.helper;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.UUID;

public class MyVectorAssember extends Transformer {

    VectorAssembler vectorAssembler;
    private String _uuid;

    public MyVectorAssember(){
        super();
        this._uuid = UUID.randomUUID().toString();
    }

    public Dataset<Row> transform(final Dataset<?> dataset){
        dataset.printSchema();
        vectorAssembler = new VectorAssembler()
                .setInputCols(dataset.drop("quality").columns())
                .setOutputCol("features");

        return vectorAssembler.transform(dataset);
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema.add("features" , new VectorUDT());
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return defaultCopy(extra);
    }

    @Override
    public String uid() {
        return this._uuid;
    }
}
