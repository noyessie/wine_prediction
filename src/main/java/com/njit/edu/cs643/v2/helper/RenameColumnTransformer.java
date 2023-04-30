package com.njit.edu.cs643.v2.helper;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.UUID;

import static com.njit.edu.cs643.helper.Data.generateNewColumnName;

public class RenameColumnTransformer extends Transformer{

    private String _uuid;

    public RenameColumnTransformer(){
        super();
        this._uuid = UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        dataset.printSchema();
        return dataset.select(generateNewColumnName(dataset.columns()));
    }

    @Override
    public StructType transformSchema(StructType schema) {

        StructType newSchema = new StructType();
        StructField[] fields = Arrays.stream(schema.fields()).map((StructField field) -> {
            String newName = field.name().replaceAll("\"","");
            return new StructField( newName, field.dataType() , true , field.metadata());
        }).toArray(size ->new StructField[size]);

        return new StructType(fields);
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
