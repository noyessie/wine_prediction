package com.njit.edu.cs643.v2.helper;


import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.Params;
import org.apache.spark.ml.param.StringArrayParam;
import org.apache.spark.ml.util.DefaultParamsReadable;
import org.apache.spark.ml.util.DefaultParamsWritable;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.plans.logical.DropColumns;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import static org.apache.spark.sql.functions.col;

public class PredictionColumnPrefixer extends MySuperTransformer {

    private String _uuid;

    private Param<String> prefixCol;


    private List<String> columnToPrefix = Arrays.asList(
            "rawPrediction",
            "probability",
            "prediction"
    );


    public PredictionColumnPrefixer(String uuid){
        super();
        this._uuid = uuid;
        this.prefixCol = new Param(this, "prefixCol", "Prefix to add to the predictions columns");
    }

    public PredictionColumnPrefixer(){
        super();
        this._uuid = "PredictionColumnPrefixer_"+UUID.randomUUID().toString();
        this.prefixCol = new Param(this, "prefixCol", "Prefix to add to the predictions columns");
    }

    private boolean shouldPrefixColumn(String name){
        return columnToPrefix.stream().filter((String col) -> name.startsWith(col)).findFirst().isPresent();
    }
    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        String prefix = getPrefixCol();

        Column newColumns[] = Arrays.stream(dataset.columns()).map(columName -> {

            if(shouldPrefixColumn(columName)){
                return col(columName).as(prefix+columName);
            }
            return col(columName);
        }).toArray(size -> new Column[size]);

        return dataset.select(newColumns);
    }


    @Override
    public StructType transformSchema(StructType schema) {

        String prefix = getPrefix();
        StructField[] fields = Arrays.stream(schema.fields()).map(field ->{
            if(shouldPrefixColumn(field.name())){
                return new StructField( prefix+field.name(), field.dataType() , true , field.metadata());
            }
            return field;
        }).toArray(size -> new StructField[size]);

        return new StructType(fields);

    }

    private String getPrefix(){
        return this.getPrefixCol();
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return defaultCopy(extra);
    }

    @Override
    public String uid() {
        return this._uuid;
    }


    // Getters
    public String getPrefixCol() { return get(prefixCol).get(); }

    // Setters
    public PredictionColumnPrefixer setPrefixCol(String p) {
        set(prefixCol , p);
        return this;
    }

    public Param prefixCol() {
        return this.prefixCol;
    }
}
