# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Linear Regression Lab
# MAGIC 
# MAGIC In the previous lesson, we predicted price using just one variable: bedrooms. Now, we want to predict price given a few other features.
# MAGIC 
# MAGIC Steps:
# MAGIC 0. Use the features: `bedrooms`, `bathrooms`, `bathrooms_na`, `minimum_nights`, and `number_of_reviews` as input to your VectorAssembler.
# MAGIC 0. Build a Linear Regression Model
# MAGIC 0. Evaluate the `RMSE` and the `R2`.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Build a linear regression model with multiple features
# MAGIC  - Compute various metrics to evaluate goodness of fit

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

filePath = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnbDF = spark.read.format("delta").load(filePath)
trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# ANSWER
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

vecAssembler = VectorAssembler(inputCols=["bedrooms", "bathrooms", "bathrooms_na", "minimum_nights", "number_of_reviews"], outputCol="features")

vecTrainDF = vecAssembler.transform(trainDF)
vecTestDF = vecAssembler.transform(testDF)

lrModel = LinearRegression(featuresCol="features", labelCol="price").fit(vecTrainDF)

predDF = lrModel.transform(vecTestDF)

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
rmse = regressionEvaluator.evaluate(predDF)
r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md Examine the coefficients for each of the variables.

# COMMAND ----------

for col, coef in zip(["bedrooms", "bathrooms", "bathrooms_na", "minimum_nights", "number_of_reviews"], lrModel.coefficients):
  print(col, coef)
  
print(f"intercept: {lrModel.intercept}")

# COMMAND ----------

# MAGIC %md ## Distributed Setting
# MAGIC 
# MAGIC Although we can quickly solve for the parameters when the data is small, the closed form solution doesn't scale well to large datasets. 
# MAGIC 
# MAGIC Spark uses the following approach to solve a linear regression problem:
# MAGIC 
# MAGIC * First, Spark tries to use matrix decomposition to solve the linear regression problem. 
# MAGIC * If it fails, Spark then uses [L-BFGS](https://spark.apache.org/docs/latest/ml-advanced.html#limited-memory-bfgs-l-bfgs) to solve for the parameters. L-BFGS is a limited-memory version of BFGS that is particularly suited to problems with very large numbers of variables. The [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) method belongs to [quasi-Newton methods](https://en.wikipedia.org/wiki/Quasi-Newton_method), which are used to either find zeroes or local maxima and minima of functions iteratively. 
# MAGIC 
# MAGIC If you are interested in how linear regression is implemented in the distributed setting and bottlenecks, check out these lecture slides:
# MAGIC * [distributed-linear-regression-1](https://files.training.databricks.com/static/docs/distributed-linear-regression-1.pdf)
# MAGIC * [distributed-linear-regression-2](https://files.training.databricks.com/static/docs/distributed-linear-regression-2.pdf)

# COMMAND ----------

# MAGIC %md ### Next Steps
# MAGIC 
# MAGIC Yikes! We built a pretty bad model. In the next notebook, we will see how we can further improve upon our model.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
