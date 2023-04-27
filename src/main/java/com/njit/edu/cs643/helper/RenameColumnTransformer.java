package com.njit.edu.cs643.helper;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.Params;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;

import static com.njit.edu.cs643.helper.Data.generateNewColumnName;

import java.util.UUID;

public class RenameColumnTransformer extends Transformer{

    private String _uuid;

    public RenameColumnTransformer(){
        super();
        this._uuid = UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        return dataset.select(generateNewColumnName(dataset.columns()));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema;
    }

    @Override
    public Transformer copy(ParamMap extra) {
        Transformer t =  new RenameColumnTransformer();
        return t;
    }

    @Override
    public String uid() {
        return this._uuid;
    }
}
