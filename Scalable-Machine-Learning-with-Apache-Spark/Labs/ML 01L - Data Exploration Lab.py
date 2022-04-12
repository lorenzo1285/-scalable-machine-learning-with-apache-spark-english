# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Data Exploration
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
# MAGIC Let's keep 80% for the training set and set aside 20% of our data for the test set. We will use the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html?highlight=randomsplit#pyspark.sql.DataFrame.randomSplit" target="_blank">randomSplit</a> method.
# MAGIC 
# MAGIC We will discuss more about the train-test split later, but throughout this notebook, do your data exploration on **`train_df`**.

# COMMAND ----------

file_path = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's make a histogram of the price column to explore it (change the number of bins to 300).  

# COMMAND ----------

display(train_df.select("price"))

# COMMAND ----------

# MAGIC %md
# MAGIC Is this a <a href="https://en.wikipedia.org/wiki/Log-normal_distribution" target="_blank">Log Normal</a> distribution? Take the **`log`** of price and check the histogram. Keep this in mind for later :).

# COMMAND ----------

# TODO

display(<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC Now take a look at how **`price`** depends on some of the variables:
# MAGIC * Plot **`price`** vs **`bedrooms`**
# MAGIC * Plot **`price`** vs **`accommodates`**
# MAGIC 
# MAGIC Make sure to change the aggregation to **`AVG`**.

# COMMAND ----------

display(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the distribution of some of our categorical features

# COMMAND ----------

display(train_df.groupBy("room_type").count())

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

lat_long_price_values = train_df.select(col("latitude"), col("longitude"), col("price")/600).collect()

lat_long_price_strings = [f"[{lat}, {long}, {price}]" for lat, long, price in lat_long_price_values]

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

# MAGIC %md ## Baseline Model
# MAGIC 
# MAGIC Before we build any Machine Learning models, we want to build a baseline model to compare to. We also want to determine a metric to evaluate our model. Let's use RMSE here.
# MAGIC 
# MAGIC For this dataset, let's build a baseline model that always predicts the average price and one that always predicts the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.approxQuantile.html?highlight=approxquantile#pyspark.sql.DataFrame.approxQuantile" target="_blank">median</a> price, and see how we do. Do this in two separate steps:
# MAGIC 
# MAGIC 0. **`train_df`**: Extract the average and median price from **`train_df`**, and store them in the variables **`avg_price`** and **`median_price`**, respectively.
# MAGIC 0. **`test_df`**: Create two additional columns called **`avgPrediction`** and **`medianPrediction`** with the average and median price from **`train_df`**, respectively. Call the resulting DataFrame **`pred_df`**. 
# MAGIC 
# MAGIC Some useful functions:
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.avg.html?highlight=avg#pyspark.sql.functions.avg" target="_blank">avg()</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.col.html?highlight=col#pyspark.sql.functions.col" target="_blank">col()</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.lit.html?highlight=lit#pyspark.sql.functions.lit" target="_blank">lit()</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.approxQuantile.html?highlight=approxquantile#pyspark.sql.DataFrame.approxQuantile" target="_blank">approxQuantile()</a> **HINT**: There is no median function, so you will need to use approxQuantile
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.withColumn.html?highlight=withcolumn#pyspark.sql.DataFrame.withColumn" target="_blank">withColumn()</a>

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md ## Evaluate model
# MAGIC 
# MAGIC We are going to use SparkML's <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RegressionEvaluator.html?highlight=regressionevaluator#pyspark.ml.evaluation.RegressionEvaluator" target="_blank">RegressionEvaluator</a> to compute the <a href="https://en.wikipedia.org/wiki/Root-mean-square_deviation" target="_blank">root mean square error (RMSE)</a>. 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> We'll discuss the specifics of the RMSE evaluation metric along with SparkML's evaluators in the next lesson. For now, simply realize that RMSE quantifies how far off our predictions are from the true values on average.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_mean_evaluator = RegressionEvaluator(predictionCol="avgPrediction", labelCol="price", metricName="rmse")
print(f"The RMSE for predicting the average price is: {regression_mean_evaluator.evaluate(pred_df)}")

regressionMedianEvaluator = RegressionEvaluator(predictionCol="medianPrediction", labelCol="price", metricName="rmse")
print(f"The RMSE for predicting the median price is: {regressionMedianEvaluator.evaluate(pred_df)}")

# COMMAND ----------

# MAGIC %md Wow! We can see that always predicting median or mean doesn't do too well for our dataset. Let's see if we can improve this with a machine learning model!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
