# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Linear Regression: Improving our model
# MAGIC 
# MAGIC In this notebook we will be adding additional features to our model, as well as discuss how to handle categorical features.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - One Hot Encode categorical variables
# MAGIC  - Use the Pipeline API
# MAGIC  - Save and load models

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

filePath = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnbDF = spark.read.format("delta").load(filePath)

# COMMAND ----------

# MAGIC %md ## Train/Test Split
# MAGIC 
# MAGIC Let's use the same 80/20 split with the same seed as the previous notebook so we can compare our results apples to apples (unless you changed the cluster config!)

# COMMAND ----------

trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md ## Categorical Variables
# MAGIC 
# MAGIC There are a few ways to handle categorical features:
# MAGIC * Assign them a numeric value
# MAGIC * Create "dummy" variables (also known as One Hot Encoding)
# MAGIC * Generate embeddings (mainly used for textual data)
# MAGIC 
# MAGIC ### One Hot Encoder
# MAGIC Here, we are going to One Hot Encode (OHE) our categorical variables. Spark doesn't have a `dummies` function, and OHE is a two step process. First, we need to use [StringIndexer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html?highlight=stringindexer#pyspark.ml.feature.StringIndexer) to map a string column of labels to an ML column of label indices.
# MAGIC 
# MAGIC Then, we can apply the [OneHotEncoder](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html?highlight=onehotencoder#pyspark.ml.feature.OneHotEncoder) to the output of the StringIndexer.

# COMMAND ----------

# MAGIC %md 
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC In Spark 2.3, the StringIndexer only takes 1 column as input, but in 3.0, it accepts multiple columns as input.
# MAGIC 
# MAGIC Another change in Spark 3.0 is the OneHotEncoderEstimator was renamed to OneHotEncoder.

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]
oheOutputCols = [x + "OHE" for x in categoricalCols]

stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="skip")
oheEncoder = OneHotEncoder(inputCols=indexOutputCols, outputCols=oheOutputCols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Assembler
# MAGIC 
# MAGIC Now we can combine our OHE categorical features with our numeric features.

# COMMAND ----------

# MAGIC %md 
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC A lot of students will forget to remove the label from their features. This is why we explicitly exclude price.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == "double") & (field != "price"))]
assemblerInputs = oheOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md ## Linear Regression
# MAGIC 
# MAGIC Now that we have all of our features, let's build a linear regression model.

# COMMAND ----------

# MAGIC %md 
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC We are cloning the stages in the pipeline in case students accidentally run the cell multiple times (that would result in duplicate stages).

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="price", featuresCol="features")

# COMMAND ----------

# MAGIC %md ## Pipeline
# MAGIC 
# MAGIC Let's put all these stages in a Pipeline. A [Pipeline](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html?highlight=pipeline#pyspark.ml.Pipeline) is a way of organizing all of our transformers and estimators.
# MAGIC 
# MAGIC This way, we don't have to worry about remembering the same ordering of transformations to apply to our test dataset.

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [stringIndexer, oheEncoder, vecAssembler, lr]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md ## Saving Models
# MAGIC 
# MAGIC We can save our models to persistent storage (e.g. DBFS) in case our cluster goes down so we don't have to recompute our results.

# COMMAND ----------

# MAGIC %md 
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC We recommend that students ALWAYS use Pipelines when they save their models. That way they don't have to guess what type of model they are loading back in (e.g. was it a DecisionTreeClassifier?). You can also use Pipelines as stages within Pipelines.

# COMMAND ----------

pipelinePath = userhome + "/machine-learning/lr_pipeline_model"
pipelineModel.write().overwrite().save(pipelinePath)

# COMMAND ----------

# MAGIC %md ## Loading models
# MAGIC 
# MAGIC When you load in models, you need to know the type of model you are loading back in (was it a linear regression or logistic regression model?).
# MAGIC 
# MAGIC For this reason, we recommend you always put your transformers/estimators into a Pipeline, so you can always load the generic PipelineModel back in.

# COMMAND ----------

from pyspark.ml import PipelineModel

savedPipelineModel = PipelineModel.load(pipelinePath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply model to test set

# COMMAND ----------

predDF = savedPipelineModel.transform(testDF)

display(predDF.select("features", "price", "prediction"))

# COMMAND ----------

# MAGIC %md ## Evaluate model
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/r2d2.jpg) How is our R2 doing? 

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regressionEvaluator.evaluate(predDF)
r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md As you can see, our RMSE decreased when compared to the model without one-hot encoding, and the R2 increased as well!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
