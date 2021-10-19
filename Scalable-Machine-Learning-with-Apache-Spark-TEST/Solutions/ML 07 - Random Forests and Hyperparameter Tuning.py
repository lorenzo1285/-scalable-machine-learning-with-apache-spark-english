# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Random Forests and Hyperparameter Tuning
# MAGIC 
# MAGIC Now let's take a look at how to tune random forests using grid search and cross validation in order to find the optimal hyperparameters.  Using the Databricks Runtime for ML, MLflow automatically logs your experiments with the SparkML cross-validator!
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Tune hyperparameters using Grid Search
# MAGIC  - Optimize a SparkML pipeline

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

filePath = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnbDF = spark.read.format("delta").load(filePath)
trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)

categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]

stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="skip")

numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == "double") & (field != "price"))]
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

rf = RandomForestRegressor(labelCol="price", maxBins=40)
stages = [stringIndexer, vecAssembler, rf]
pipeline = Pipeline(stages=stages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ParamGrid
# MAGIC 
# MAGIC First let's take a look at the various hyperparameters we could tune for random forest.
# MAGIC 
# MAGIC **Pop quiz:** what's the difference between a parameter and a hyperparameter?

# COMMAND ----------

print(rf.explainParams())

# COMMAND ----------

# MAGIC %md There are a lot of hyperparameters we could tune, and it would take a long time to manually configure.
# MAGIC 
# MAGIC Instead of a manual (ad-hoc) approach, let's use Spark's [ParamGridBuilder](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html?highlight=paramgridbuilder#pyspark.ml.tuning.ParamGridBuilder) to find the optimal hyperparameters in a more systematic approach.
# MAGIC 
# MAGIC Let's define a grid of hyperparameters to test:
# MAGIC   - `maxDepth`: max depth of each decision tree (Use the values `2, 5`)
# MAGIC   - `numTrees`: number of decision trees to train (Use the values `5, 10`)
# MAGIC 
# MAGIC `addGrid()` accepts the name of the parameter (e.g. `rf.maxDepth`), and a list of the possible values (e.g. `[2, 5]`).

# COMMAND ----------

# MAGIC %md
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC Deeper trees are more expressive (potentially allowing higher accuracy), but they are also more costly to train and are more likely to overfit.
# MAGIC 
# MAGIC We chose the hyperparam values because they train relatively quickly.
# MAGIC 
# MAGIC NOTE: Performance degrades exponentially as you grow deeper trees, and SparkML has a cutoff around depth of 30.

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

paramGrid = (ParamGridBuilder()
            .addGrid(rf.maxDepth, [2, 5])
            .addGrid(rf.numTrees, [5, 10])
            .build())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross Validation
# MAGIC 
# MAGIC We are also going to use 3-fold cross validation to identify the optimal hyperparameters.
# MAGIC 
# MAGIC ![crossValidation](https://files.training.databricks.com/images/301/CrossValidation.png)
# MAGIC 
# MAGIC With 3-fold cross-validation, we train on 2/3 of the data, and evaluate with the remaining (held-out) 1/3. We repeat this process 3 times, so each fold gets the chance to act as the validation set. We then average the results of the three rounds.

# COMMAND ----------

# MAGIC %md
# MAGIC We pass in the `estimator` (pipeline), `evaluator`, and `estimatorParamMaps` to [CrossValidator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator) so that it knows:
# MAGIC - Which model to use
# MAGIC - How to evaluate the model
# MAGIC - What hyperparameters to set for the model
# MAGIC 
# MAGIC We can also set the number of folds we want to split our data into (3), as well as setting a seed so we all have the same split in the data.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")

cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, 
                    numFolds=3, seed=42)

# COMMAND ----------

# MAGIC %md **Question**: How many models are we training right now?

# COMMAND ----------

# MAGIC %md
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC We are training 13 models: 3-fold cross validation combined with 2 different values for the `maxDepth` and 2 for `numTrees`, plus one to retrain on all the training data.
# MAGIC 
# MAGIC SparkML automatically retrains on all the data because otherwise you wouldn't be able to combine those three submodels trained on different folds of the data.

# COMMAND ----------

cvModel = cv.fit(trainDF)

# COMMAND ----------

# MAGIC %md ## Parallelism Parameter
# MAGIC 
# MAGIC Hmmm... that took a long time to run. That's because the models were being trained sequentially rather than in parallel!
# MAGIC 
# MAGIC In Spark 2.3, a [parallelism](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator.parallelism) parameter was introduced. From the docs: `the number of threads to use when running parallel algorithms (>= 1)`.
# MAGIC 
# MAGIC Let's set this value to 4 and see if we can train any faster. The Spark [docs](https://spark.apache.org/docs/latest/ml-tuning.html) recommend a value between 2-10.

# COMMAND ----------

# MAGIC %md
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC The value to set parallelism depends on the size of your model and the size of your cluster. Some models, like linear regression, are very cheap to store + relatively cheap to compute.
# MAGIC 
# MAGIC However, some models like ALS are distributed models stored across the workers rather than the driver.
# MAGIC 
# MAGIC Parallelism is like a hyperparameter - try a few to find the best value to speed up your Spark jobs. We recommend trying 2, 4, and 8.

# COMMAND ----------

cvModel = cv.setParallelism(4).fit(trainDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC **Question**: Hmmm... that still took a long time to run. Should we put the pipeline in the cross validator, or the cross validator in the pipeline?
# MAGIC 
# MAGIC It depends if there are estimators or transformers in the pipeline. If you have things like StringIndexer (an estimator) in the pipeline, then you have to refit it every time if you put the entire pipeline in the cross validator.
# MAGIC 
# MAGIC However, if there is any concern about data leakage from the earlier steps, the safest thing is to put the pipeline inside the CV, not the other way. CV first splits the data and then .fit() the pipeline. If it is placed at the end of the pipeline, we potentially can leak the info from hold-out set to train set.

# COMMAND ----------

# MAGIC %md
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC If your pipeline only contains transformers, there won't be a notable speed difference.

# COMMAND ----------

cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=paramGrid, 
                    numFolds=3, parallelism=4, seed=42)

stagesWithCV = [stringIndexer, vecAssembler, cv]
pipeline = Pipeline(stages=stagesWithCV)

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the model with the best hyperparameter configuration

# COMMAND ----------

list(zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics))

# COMMAND ----------

predDF = pipelineModel.transform(testDF)

rmse = evaluator.evaluate(predDF)
r2 = evaluator.setMetricName("r2").evaluate(predDF)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md Progress!  Looks like we're out-performing decision trees.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
