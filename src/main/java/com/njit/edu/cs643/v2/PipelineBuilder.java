package com.njit.edu.cs643.v2;

import com.njit.edu.cs643.v2.helper.ColumnWeightTransformer;
import com.njit.edu.cs643.v2.helper.MyVectorAssember;
import com.njit.edu.cs643.v2.helper.PredictionColumnPrefixer;
import com.njit.edu.cs643.v2.helper.RenameColumnTransformer;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;

import java.util.ArrayList;
import java.util.List;

public class PipelineBuilder {

    private Pipeline pipeline;
    private ParamMap[] params;

    public PipelineBuilder(){
        this.pipeline = new Pipeline();
        List<PipelineStage> stages = new ArrayList<>();

        ParamGridBuilder builder = new ParamGridBuilder(); // for optimization, we will be setting the params to optimize there.

        //we rename the columns to the right name
        RenameColumnTransformer renameColums = new RenameColumnTransformer();
        stages.add(renameColums);


        //assemble features
        MyVectorAssember featureAssembler = new MyVectorAssember();
        stages.add(featureAssembler);

        //adding weight to prediction so we can improove the logistic regression algorithm
        Transformer addWeights = new ColumnWeightTransformer();
        stages.add(addWeights);



        //we can now add the models
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("features")
                .setFamily("multinomial")
                .setMaxIter(10)
                .setRegParam(0.1)
                .setElasticNetParam(0.3)
                .setFamily("multinomial")
                .setWeightCol("weight");



        //params for logistic regression
        builder.addGrid(lr.maxIter() , new int[]{500,100})
                .addGrid(lr.elasticNetParam() , new double[]{0.1})
                .build();
        stages.add(lr);
        stages.add(new PredictionColumnPrefixer().setPrefixCol("lr_"));


        // Random Forest Classifier
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("quality")
                .setFeaturesCol("features");
        builder.addGrid(rf.minInfoGain() , new double[]{0.1})
                .addGrid(rf.maxDepth() , new int[]{15 , 20, 30})
                .addGrid(rf.numTrees() , new int[]{50 , 100 , 250});
        stages.add(rf);
        stages.add(new PredictionColumnPrefixer().setPrefixCol("rf_"));

        //adding MLP prediction
        MultilayerPerceptronClassifier mlp = new MultilayerPerceptronClassifier()
                .setLabelCol("quality")
                .setFeaturesCol("features")
                .setLayers(new int[]{11,20,25,20,10})
                .setBlockSize(128);
        builder.addGrid(mlp.maxIter() , new int[]{500,1000})
                .addGrid(mlp.blockSize() , new int[]{100});
        stages.add(mlp);
        stages.add(new PredictionColumnPrefixer().setPrefixCol("mlp_"));

        //Now we merge the 3 result into one pipeline to see if we can get better result
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"lr_prediction" , "rf_prediction","mlp_prediction"})
                .setOutputCol("composite_feature");
        stages.add(assembler);

        LogisticRegression lgr = new LogisticRegression();
            lgr = lgr.setLabelCol("quality")
                    .setFeaturesCol("composite_feature")
                    .setMaxIter(250)
                    .setFamily("multinomial")
                    .setWeightCol("weight");
            stages.add(lgr);

        this.params = builder.build();

        pipeline.setStages(stages.stream().toArray(size -> new PipelineStage[size]));
    }

    public Pipeline getPipeline(){
        return pipeline;
    }

    public ParamMap[] getParams(){
        return params;
    }
}
