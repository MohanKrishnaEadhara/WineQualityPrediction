package org.winequality;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPrediction {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("WineQualityPrediction");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().appName("WineQualityPrediction").getOrCreate();

        try {
            // Load the saved model
            PipelineModel model = PipelineModel.load("s3://wine-quality-dataset-bucket/WineQualityModel");

            // Load the test dataset
            Dataset<Row> testData = spark.read().format("csv")
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .load("s3://wine-quality-dataset-bucket/TestDataset.csv");

            // Make predictions on the test dataset
            Dataset<Row> predictions = model.transform(testData);

            // Evaluate the model performance on the test dataset
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("quality")
                    .setPredictionCol("prediction")
                    .setMetricName("f1");

            // Print the F1 score on the test dataset
            double f1Score = evaluator.evaluate(predictions);
            System.out.println("F1 score on test data: " + f1Score);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            jsc.close();
        }
    }
}
