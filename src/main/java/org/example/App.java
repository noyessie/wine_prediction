package org.example;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        SparkConf conf  = new SparkConf().setAppName("CS 643 Programming Assigment 2")
                .setMaster("local[2]");

        JavaSparkContext sc = new JavaSparkContext(conf);

        String dataFile = "data/training.csv";

        JavaRDD<String> data = sc.textFile(dataFile);
//
//        long c = data.mapPartitionsWithIndex((index , iterator)-> {
//            System.out.println("+=======================================================");
//            System.out.println("Index " + index);
//            while(iterator.hasNext()){
//                System.out.println("Iterator " + iterator.next());
//            }
//            return iterator;
//        }, false).count();

        final String header = data.first();
        JavaRDD<LabeledPoint> inputDataWithLabel = data.filter(line -> true).map(line -> {
            if(!header.equalsIgnoreCase(line)){
                String[] parts = line.split(";");
                double[] v = new double[parts.length -1];
                //System.out.println("data "+ parts.length);
                for(int i = 0; i< parts.length-1 ; i++){
                    v[i] = Double.parseDouble(parts[i]);
                }
                return new LabeledPoint(Double.parseDouble(parts[parts.length-1]) , Vectors.dense(v));
            }
            return null;
        }).filter(vector -> vector != null);

//        Vector vector = inputData.rdd().first();
//        System.out.println(vector.toArray());
//
//        MultivariateStatisticalSummary summary = Statistics.colStats(inputData.rdd());
//
//        System.out.println("Summary Mean: ");
//        System.out.println(summary.mean());
//
//        System.out.println("Summary Variance : ");
//        System.out.println(summary.variance());

        //Split data
        JavaRDD<LabeledPoint>[] splits = inputDataWithLabel.randomSplit(new double[]{0.8 , 0.2});

        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];


        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        LogisticRegressionModel model = new LogisticRegressionWithLBFGS().setNumClasses(10).run(trainingData.rdd());


        JavaPairRDD<Object , Object> predictionAndLabels = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()) , p.label()));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

        double accuracy = metrics.accuracy();

        System.out.println("Model Accuracy on test Data: "+ accuracy);

        //LogisticRegressionModel lrModel = lr.fit(trainingData.d);

        //saving the model
        //model.save(sc.con , "model/logistic-regression");

    }
}
