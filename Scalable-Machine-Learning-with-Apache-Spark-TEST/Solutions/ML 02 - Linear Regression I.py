# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Regression: Predicting Rental Price
# MAGIC 
# MAGIC In this notebook, we will use the dataset we cleansed in the previous lab to predict Airbnb rental prices in San Francisco.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use the SparkML API to build a linear regression model
# MAGIC  - Identify the differences between estimators and transformers

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

filePath = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnbDF = spark.read.format("delta").load(filePath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/301/TrainTestSplit.png)
# MAGIC 
# MAGIC **Question**: Why is it necessary to set a seed? What happens if I change my cluster configuration?

# COMMAND ----------

# MAGIC %md 
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC Below, we show what happens if you "change" your cluster config by repartitioning your data. To test this out, try spinning up a cluster with just one worker, and another with two workers. 
# MAGIC 
# MAGIC When you do an 80/20 train/test split, it is an "approximate" 80/20 split. It is not an exact 80/20 split, and when we repartition the data (aka change the cluster config), we show that we get not only a different # of data points in train/test, but also different data points.
# MAGIC 
# MAGIC Our recommendation is to split your data once, then write it out to its own train/test folder so you don't have these reproducibility issues.
# MAGIC 
# MAGIC Example:
# MAGIC 
# MAGIC 1- Let's say you have a dataset: {1,3,2,4,5,6,9,7, 8,10}.
# MAGIC 2- You do 0.8 0.2 split
# MAGIC 3- Let's say Spark decides to create two partition of the data when you run .randomSplit()
# MAGIC 4- partition1 {1,4,3,6,10}. partition2 {2,7,5,9,8}
# MAGIC 5- Spark orders each partition {1,3,4,6,10} and {2,5,7,8,9}
# MAGIC 6- If you use random with replacement spark uses Poisson sampler otherwise it uses Bernoulli sampler
# MAGIC 7- Spark creates random number for each element
# MAGIC {1:0.24,3:0.65,4:0.7,6:0.82,10:0.54} and {2:0.14,5:0.98,7:0.33,8:0.76,9:0.87}
# MAGIC 8- Spark picks those numbers that are associated with <0.8 (since we are doing 80-20 split)
# MAGIC 9- Spark picks {1,3,4,10} from partition 1 and {2,7,8} from partition 2
# MAGIC 
# MAGIC Now if you run the command again, spark "reshuffles the data". There is no guarantee that each partition gets same number of elements. For example:
# MAGIC 
# MAGIC 1- partition1 {1,5,3,4}. partition2 {6,2,7,5,9,8, 10}
# MAGIC 2- Spark orders each partition {1,3,4,5} and {2,6,7,8,9,10}
# MAGIC 3- If you use random with replacement spark uses Poisson sampler otherwise it uses Bernoulli sampler
# MAGIC 4- Spark creates random number for each element (if you use the same seed, it will be the same order of random numbers.)
# MAGIC {1:0.24,3:0.65,4:0.7,5:0.87} and {2:0.14,6:0.98,7:0.33,8:0.76,9:0.87,10:0.98}
# MAGIC 5- Spark picks those numbers that are associated with <0.8 (we are doing 80-20 split)
# MAGIC 6- Spark picks {1,3,4} from partition 1 and {2,7,8} from partition 2
# MAGIC 
# MAGIC So we ended up with 7 elements when we ran randomSplit() first time and ended up with 6 elements when we ran randomSplit() the second time.
# MAGIC 
# MAGIC Same thing happens in our notebook. We actually do not need to do .repartition(). Even without repartitioning, if we do not cache() the data, we might end up with different elements in our random split.

# COMMAND ----------

trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)
print(trainDF.cache().count())

# COMMAND ----------

# MAGIC %md
# MAGIC Let's change the # of partitions (to simulate a different cluster configuration), and see if we get the same number of data points in our training set. 

# COMMAND ----------

trainRepartitionDF, testRepartitionDF = (airbnbDF
                                         .repartition(24)
                                         .randomSplit([.8, .2], seed=42))

print(trainRepartitionDF.count())

# COMMAND ----------

# MAGIC %md ## Linear Regression
# MAGIC 
# MAGIC We are going to build a very simple model predicting `price` just given the number of `bedrooms`.
# MAGIC 
# MAGIC **Question**: What are some assumptions of the linear regression model?

# COMMAND ----------

# MAGIC %md 
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC * Linear relationship between X & y
# MAGIC * Errors are normally distributed (Homoscedasticity)
# MAGIC * Features are independent

# COMMAND ----------

display(trainDF.select("price", "bedrooms"))

# COMMAND ----------

display(trainDF.select("price", "bedrooms").summary())

# COMMAND ----------

display(trainDF)

# COMMAND ----------

# MAGIC %md There do appear some outliers in our dataset for the price ($10,000 a night??). Just keep this in mind when we are building our models :).
# MAGIC 
# MAGIC We will use [LinearRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html?highlight=linearregression#pyspark.ml.regression.LinearRegression) to build our first model.
# MAGIC 
# MAGIC The cell below will fail because the Linear Regression estimator expects a vector of values as input. We will fix that with VectorAssembler below. 

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="bedrooms", labelCol="price")

# Uncomment when running
# lrModel = lr.fit(trainDF)

# COMMAND ----------

# MAGIC %md ## Vector Assembler
# MAGIC 
# MAGIC What went wrong? Turns out that the Linear Regression **estimator** (`.fit()`) expected a column of Vector type as input.
# MAGIC 
# MAGIC We can easily get the values from the `bedrooms` column into a single vector using [VectorAssembler](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler). VectorAssembler is an example of a **transformer**. Transformers take in a DataFrame, and return a new DataFrame with one or more columns appended to it. They do not learn from your data, but apply rule based transformations.
# MAGIC 
# MAGIC You can see an example of how to use VectorAssembler on the [ML Programming Guide](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler).

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")

vecTrainDF = vecAssembler.transform(trainDF)

# COMMAND ----------

lr = LinearRegression(featuresCol="features", labelCol="price")
lrModel = lr.fit(vecTrainDF)

# COMMAND ----------

# MAGIC %md ## Inspect the model

# COMMAND ----------

m = lrModel.coefficients[0]
b = lrModel.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply model to test set

# COMMAND ----------

# MAGIC %md 
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC We are using `.show()` instead of `display()` below because display in Databricks is a bit funky when working with vectors. More on this in the next notebook.

# COMMAND ----------

vecTestDF = vecAssembler.transform(testDF)

predDF = lrModel.transform(vecTestDF)

predDF.select("bedrooms", "features", "price", "prediction").show()

# COMMAND ----------

# MAGIC %md ## Evaluate Model
# MAGIC 
# MAGIC Let's see how our linear regression model with just one variable does. Does it beat our baseline model?

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regressionEvaluator.evaluate(predDF)
print(f"RMSE is {rmse}")

# COMMAND ----------

# MAGIC %md Wahoo! Our RMSE is better than our baseline model. However, it's still not that great. Let's see how we can further decrease it in future notebooks.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
