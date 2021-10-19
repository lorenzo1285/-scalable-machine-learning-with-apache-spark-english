# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Hyperparameter Tuning with Random Forests
# MAGIC 
# MAGIC In this lab, you will convert the Airbnb problem to a classification dataset, build a random forest classifier, and tune some hyperparameters of the random forest.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Perform grid search on a random forest
# MAGIC  - Generate feature importance scores and classification metrics
# MAGIC  - Identify differences between scikit-learn's Random Forest and SparkML's
# MAGIC  
# MAGIC You can read more about the distributed implementation of Random Forests in the Spark [source code](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/tree/impl/RandomForest.scala#L42).

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md ## From Regression to Classification
# MAGIC 
# MAGIC In this case, we'll turn the Airbnb housing dataset into a classification problem to **classify between high and low price listings.**  Our `class` column will be:<br><br>
# MAGIC 
# MAGIC - `0` for a low cost listing of under $150
# MAGIC - `1` for a high cost listing of $150 or more

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

filePath = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"

airbnbDF = (spark.read.format("delta").load(filePath)
  .withColumn("priceClass", (col("price") >= 150).cast("int"))
  .drop("price")
)

trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)

categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]

stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="skip")

numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == "double") & (field != "priceClass"))]
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md ## Why can't we OHE?
# MAGIC 
# MAGIC **Question:** What would go wrong if we One Hot Encoded our variables before passing them into the random forest?
# MAGIC 
# MAGIC **HINT:** Think about what would happen to the "randomness" of feature selection.

# COMMAND ----------

# MAGIC %md ## Random Forest
# MAGIC 
# MAGIC Create a Random Forest classifer called `rf` with the `labelCol`=`priceClass`, `maxBins`=`40`, and `seed`=`42` (for reproducibility).
# MAGIC 
# MAGIC It's under `pyspark.ml.classification.RandomForestClassifier` in Python.

# COMMAND ----------

# ANSWER
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="priceClass", maxBins=40, seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grid Search
# MAGIC 
# MAGIC There are a lot of hyperparameters we could tune, and it would take a long time to manually configure.
# MAGIC 
# MAGIC Let's use Spark's [ParamGridBuilder](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html?highlight=paramgrid#pyspark.ml.tuning.ParamGridBuilder) to find the optimal hyperparameters in a more systematic approach.
# MAGIC 
# MAGIC Let's define a grid of hyperparameters to test:
# MAGIC   - maxDepth: max depth of the decision tree (Use the values `2, 5, 10`)
# MAGIC   - numTrees: number of decision trees (Use the values `10, 20, 100`)
# MAGIC 
# MAGIC `addGrid()` accepts the name of the parameter (e.g. `rf.maxDepth`), and a list of the possible values (e.g. `[2, 5, 10]`).

# COMMAND ----------

# ANSWER

from pyspark.ml.tuning import ParamGridBuilder

paramGrid = (ParamGridBuilder()
            .addGrid(rf.maxDepth, [2, 5, 10])
            .addGrid(rf.numTrees, [10, 20, 100])
            .build())

# COMMAND ----------

# MAGIC %md ## Evaluator
# MAGIC 
# MAGIC In the past, we used a `RegressionEvaluator`.  For classification, we can use a [BinaryClassificationEvaluator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html?highlight=binaryclass#pyspark.ml.evaluation.BinaryClassificationEvaluator) if we have two classes or [MulticlassClassificationEvaluator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.MulticlassClassificationEvaluator.html?highlight=multiclass#pyspark.ml.evaluation.MulticlassClassificationEvaluator) for more than two classes.
# MAGIC 
# MAGIC Create a `BinaryClassificationEvaluator` with `areaUnderROC` as the metric.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> [Read more on ROC curves here.](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)  In essence, it compares true positive and false positives.

# COMMAND ----------

# ANSWER
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="priceClass")

# COMMAND ----------

# MAGIC %md ## Cross Validation
# MAGIC 
# MAGIC We are going to do 3-Fold cross-validation, with `parallelism`=4, and set the `seed`=42 on the cross-validator for reproducibility.
# MAGIC 
# MAGIC Put the Random Forest in the CV to speed up the [cross validation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator) (as opposed to the pipeline in the CV).

# COMMAND ----------

# ANSWER
from pyspark.ml.tuning import CrossValidator

cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=paramGrid,
                    numFolds=3, parallelism=4, seed=42)

# COMMAND ----------

# MAGIC %md ## Pipeline
# MAGIC 
# MAGIC Let's fit the pipeline with our cross validator to our training data (this may take a few minutes).

# COMMAND ----------

stages = [stringIndexer, vecAssembler, cv]

pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md ## Hyperparameter
# MAGIC 
# MAGIC Which hyperparameter combination performed the best?

# COMMAND ----------

cvModel = pipelineModel.stages[-1]
rfModel = cvModel.bestModel

# list(zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics))

print(rfModel.explainParams())

# COMMAND ----------

# MAGIC %md ## Feature Importance

# COMMAND ----------

import pandas as pd

pandasDF = pd.DataFrame(list(zip(vecAssembler.getInputCols(), rfModel.featureImportances)), columns=["feature", "importance"])
topFeatures = pandasDF.sort_values(["importance"], ascending=False)
topFeatures

# COMMAND ----------

# MAGIC %md Do those features make sense? Would you use those features when picking an Airbnb rental?

# COMMAND ----------

# MAGIC %md ## Apply Model to test set

# COMMAND ----------

# ANSWER

predDF = pipelineModel.transform(testDF)
areaUnderROC = evaluator.evaluate(predDF)
print(f"Area under ROC is {areaUnderROC:.2f}")

# COMMAND ----------

# MAGIC %md ## Save Model
# MAGIC 
# MAGIC Save the model to `<userhome>/machine-learning/rf_pipeline_model`.

# COMMAND ----------

# ANSWER
pipelineModel.write().overwrite().save(userhome + "/machine-learning/rf_pipeline_model")

# COMMAND ----------

# MAGIC %md ## Sklearn vs SparkML
# MAGIC 
# MAGIC [Sklearn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) vs [SparkML RandomForestRegressor](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.RandomForestRegressor.html?highlight=randomfore#pyspark.ml.regression.RandomForestRegressor).
# MAGIC 
# MAGIC Look at these params in particular:
# MAGIC * **n_estimators** (sklearn) vs **numTrees** (SparkML)
# MAGIC * **max_depth** (sklearn) vs **maxDepth** (SparkML)
# MAGIC * **max_features** (sklearn) vs **featureSubsetStrategy** (SparkML)
# MAGIC * **maxBins** (SparkML only)
# MAGIC 
# MAGIC What do you notice that is different?

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
