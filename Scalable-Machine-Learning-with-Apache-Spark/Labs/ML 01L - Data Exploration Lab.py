# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Exploration
# MAGIC 
# MAGIC In this notebook, we will use the dataset we cleansed in the previous lab to do some Exploratory Data Analysis (EDA).
# MAGIC 
# MAGIC This will help us better understand our data to make a better model.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Identify log-normal distributions
# MAGIC  - Build a baseline model and evaluate

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC Let's keep 80% for the training set and set aside 20% of our data for the test set. We will use the [randomSplit](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html?highlight=randomsplit#pyspark.sql.DataFrame.randomSplit) method.
# MAGIC 
# MAGIC We will discuss more about the train-test split later, but throughout this notebook, do your data exploration on `trainDF`.

# COMMAND ----------

filePath = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnbDF = spark.read.format("delta").load(filePath)
trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's make a histogram of the price column to explore it (change the number of bins to 300).  

# COMMAND ----------

display(trainDF.select("price"))

# COMMAND ----------

# MAGIC %md
# MAGIC Is this a <a href="https://en.wikipedia.org/wiki/Log-normal_distribution" target="_blank">Log Normal</a> distribution? Take the `log` of price and check the histogram. Keep this in mind for later :).

# COMMAND ----------

# TODO

display(<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC Now take a look at how `price` depends on some of the variables:
# MAGIC * Plot `price` vs `bedrooms`
# MAGIC * Plot `price` vs `accommodates`
# MAGIC 
# MAGIC Make sure to change the aggregation to `AVG`.

# COMMAND ----------

display(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the distribution of some of our categorical features

# COMMAND ----------

display(trainDF.groupBy("room_type").count())

# COMMAND ----------

# MAGIC %md
# MAGIC Which neighbourhoods have the highest number of rentals? Display the neighbourhoods and their associated count in descending order.

# COMMAND ----------

# TODO
display(<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### How much does the price depend on the location?
# MAGIC 
# MAGIC We can use displayHTML to render any HTML, CSS, or JavaScript code.

# COMMAND ----------

from pyspark.sql.functions import col

lat_long_price_values = trainDF.select(col("latitude"), col("longitude"), col("price")/600).collect()

lat_long_price_strings = [
  "[{}, {}, {}]".format(lat, long, price) 
  for lat, long, price in lat_long_price_values
]

v = ",\n".join(lat_long_price_strings)

# DO NOT worry about what this HTML code is doing! We took it from Stack Overflow :-)
displayHTML("""
<html>
<head>
 <link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.1/dist/leaflet.css"
   integrity="sha512-Rksm5RenBEKSKFjgI3a41vrjkw4EVPlJ3+OiI65vTjIdo9brlAacEuKOiQ5OFh7cOI1bkDwLqdLw3Zg0cRJAAQ=="
   crossorigin=""/>
 <script src="https://unpkg.com/leaflet@1.3.1/dist/leaflet.js"
   integrity="sha512-/Nsx9X4HebavoBvEBuyp3I7od5tA0UzAxs+j83KgC8PU0kgB4XiK4Lfe4y4cgBtaRJQEIFCW+oC506aPT2L1zw=="
   crossorigin=""></script>
 <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.heat/0.2.0/leaflet-heat.js"></script>
</head>
<body>
    <div id="mapid" style="width:700px; height:500px"></div>
  <script>
  var mymap = L.map('mapid').setView([37.7587,-122.4486], 12);
  var tiles = L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors',
}).addTo(mymap);
  var heat = L.heatLayer([""" + v + """], {radius: 25}).addTo(mymap);
  </script>
  </body>
  </html>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Model
# MAGIC 
# MAGIC Before we build any Machine Learning models, we want to build a baseline model to compare to. We also want to determine a metric to evaluate our model. Let's use RMSE here.
# MAGIC 
# MAGIC For this dataset, let's build a baseline model that always predict the average price and one that always predicts the [median](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.approxQuantile.html?highlight=approxquantile#pyspark.sql.DataFrame.approxQuantile) price, and see how we do. Do this in two separate steps:
# MAGIC 
# MAGIC 0. `trainDF`: Extract the average and median price from `trainDF`, and store them in the variables `avgPrice` and `medianPrice`, respectively.
# MAGIC 0. `testDF`: Create two additional columns called `avgPrediction` and `medianPrediction` with the average and median price from `trainDF`, respectively. Call the resulting DataFrame `predDF`. 
# MAGIC 
# MAGIC Some useful functions:
# MAGIC * [avg()](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.avg.html?highlight=avg#pyspark.sql.functions.avg)
# MAGIC * [col()](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.col.html?highlight=col#pyspark.sql.functions.col)
# MAGIC * [lit()](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.lit.html?highlight=lit#pyspark.sql.functions.lit)
# MAGIC * [approxQuantile()](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.approxQuantile.html?highlight=approxquantile#pyspark.sql.DataFrame.approxQuantile) [**HINT**: There is no median function, so you will need to use approxQuantile]
# MAGIC * [withColumn()](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.withColumn.html?highlight=withcolumn#pyspark.sql.DataFrame.withColumn)

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate model
# MAGIC 
# MAGIC We are going to use SparkML's [RegressionEvaluator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RegressionEvaluator.html?highlight=regressionevaluator#pyspark.ml.evaluation.RegressionEvaluator) to compute the [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) for our average price and median price predictions. We will dig into evaluators in more detail in the next notebook.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regressionMeanEvaluator = RegressionEvaluator(predictionCol="avgPrediction", labelCol="price", metricName="rmse")
print(f"The RMSE for predicting the average price is: {regressionMeanEvaluator.evaluate(predDF)}")

regressionMedianEvaluator = RegressionEvaluator(predictionCol="medianPrediction", labelCol="price", metricName="rmse")
print(f"The RMSE for predicting the median price is: {regressionMedianEvaluator.evaluate(predDF)}")

# COMMAND ----------

# MAGIC %md
# MAGIC Wow! We can see that always predicting median or mean doesn't do too well for our dataset. Let's see if we can improve this with a machine learning model!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
