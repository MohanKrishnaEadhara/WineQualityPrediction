package org.winequality;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class WineQualityTraining {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("WineQualityTraining");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().appName("WineQualityTraining").getOrCreate();

        try {
            // Load the training and validation datasets
            Dataset<Row> trainingData = spark.read().format("csv")
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .load("s3://wine-quality-dataset-bucket/TrainingDataset.csv");

            Dataset<Row> validationData = spark.read().format("csv")
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .load("s3://wine-quality-dataset-bucket/ValidationDataset.csv");

            // Prepare the feature columns
            String[] featureColumns = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                    "pH", "sulphates", "alcohol"};
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureColumns)
                    .setOutputCol("features");

            // Logistic Regression model
            LogisticRegression lr = new LogisticRegression()
                    .setMaxIter(10)
                    .setRegParam(0.3)
                    .setElasticNetParam(0.8)
                    .setFamily("multinomial")
                    .setLabelCol("quality");

            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, lr});
            PipelineModel model = pipeline.fit(trainingData);

            // Make predictions on the validation dataset
            Dataset<Row> predictions = model.transform(validationData);

            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("quality")
                    .setPredictionCol("prediction")
                    .setMetricName("accuracy");

            double accuracy = evaluator.evaluate(predictions);
            System.out.println("Accuracy = " + accuracy);

            // Save the model
            try {
                model.write().overwrite().save("s3://wine-quality-dataset-bucket/WineQualityModel");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Clean up
            sc.close();
        }
    }
}
