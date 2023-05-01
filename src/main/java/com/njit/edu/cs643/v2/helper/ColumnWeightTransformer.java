package com.njit.edu.cs643.v2.helper;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.DefaultParamsReadable;
import org.apache.spark.ml.util.DefaultParamsWritable;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.UUID;

import static org.apache.spark.sql.functions.count;

public class ColumnWeightTransformer extends MySuperTransformer {

    private String _uuid;

    public ColumnWeightTransformer(){
        this._uuid = "ColumnWeightTransformer_"+ UUID.randomUUID().toString();
    }

    public ColumnWeightTransformer(String uuid){
        this._uuid = uuid;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        dataset.printSchema();
        Dataset<Row> weights = dataset.groupBy("quality")
                .agg(count("quality").as("total"))
                .selectExpr("*" , "1- (total / "+dataset.count()+") as weight")
                .select("quality","weight");

        System.out.println(dataset.count());

        weights.show();

        return dataset.join(weights , "quality");
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema.add("weight" , DataTypes.createDecimalType());
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
