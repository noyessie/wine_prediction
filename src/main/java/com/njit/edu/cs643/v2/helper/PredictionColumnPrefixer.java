package com.njit.edu.cs643.v2.helper;


import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import static org.apache.spark.sql.functions.col;

public class PredictionColumnPrefixer extends Transformer {

    private String _uuid;

    private String _prefix;

    private List<String> columnToPrefix = Arrays.asList(
            "rawPrediction",
            "probability",
            "prediction"
    );


    public PredictionColumnPrefixer(String prefix){
        super();
        this._uuid = UUID.randomUUID().toString();
        this._prefix = prefix;
    }

    private boolean shouldPrefixColumn(String name){
        return columnToPrefix.stream().filter((String col) -> name.startsWith(col)).findFirst().isPresent();
    }
    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        Column newColumns[] = Arrays.stream(dataset.columns()).map(columName -> {

            if(shouldPrefixColumn(columName)){
                return col(columName).as(this._prefix+columName);
            }
            return col(columName);
        }).toArray(size -> new Column[size]);

        return dataset.select(newColumns);
    }

    @Override
    public StructType transformSchema(StructType schema) {

        StructField[] fields = Arrays.stream(schema.fields()).map(field ->{
            if(shouldPrefixColumn(field.name())){
                return new StructField( this._prefix+field.name(), field.dataType() , true , field.metadata());
            }
            return field;
        }).toArray(size -> new StructField[size]);

        return new StructType(fields);

    }

    @Override
    public Transformer copy(ParamMap extra) {
        return defaultCopy(extra);
    }

    @Override
    public String uid() {
        return null;
    }
}
