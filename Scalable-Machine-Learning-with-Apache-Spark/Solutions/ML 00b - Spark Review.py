# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Spark Review
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Create a Spark DataFrame
# MAGIC  - Analyze the Spark UI
# MAGIC  - Cache data
# MAGIC  - Go between Pandas and Spark DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://files.training.databricks.com/images/sparkcluster.png)

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md ## Spark DataFrame

# COMMAND ----------

from pyspark.sql.functions import col, rand

df = (spark.range(1, 1000000)
      .withColumn("id", (col("id") / 1000).cast("integer"))
      .withColumn("v", rand(seed=1)))

# COMMAND ----------

# MAGIC %md Why were no Spark jobs kicked off above? Well, we didn't have to actually "touch" our data, so Spark didn't need to execute anything across the cluster.

# COMMAND ----------

display(df.sample(.001))

# COMMAND ----------

# MAGIC %md ## Views
# MAGIC 
# MAGIC How can I access this in SQL?

# COMMAND ----------

df.createOrReplaceTempView("df_temp")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM df_temp LIMIT 10

# COMMAND ----------

# MAGIC %md ## Count
# MAGIC 
# MAGIC Let's see how many records we have.

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md ## Spark UI
# MAGIC 
# MAGIC Open up the Spark UI - what are the shuffle read and shuffle write fields? The command below should give you a clue.

# COMMAND ----------

df.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md ## Cache
# MAGIC 
# MAGIC For repeated access, it will be much faster if we cache our data.

# COMMAND ----------

df.cache().count()

# COMMAND ----------

# MAGIC %md ## Re-run Count
# MAGIC 
# MAGIC Wow! Look at how much faster it is now!

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Collect Data
# MAGIC 
# MAGIC When you pull data back to the driver  (e.g. call **`.collect()`**, **`.toPandas()`**,  etc), you'll need to be careful of how much data you're bringing back. Otherwise, you might get OOM exceptions!
# MAGIC 
# MAGIC A best practice is explicitly limit the number of records, unless you know your data set is small, before calling **`.collect()`** or **`.toPandas()`**.

# COMMAND ----------

df.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md ## What's new in <a href="https://www.youtube.com/watch?v=l6SuXvhorDY&feature=emb_logo" target="_blank">Spark 3.0</a>
# MAGIC 
# MAGIC * <a href="https://www.youtube.com/watch?v=jzrEc4r90N8&feature=emb_logo" target="_blank">Adaptive Query Execution</a>
# MAGIC   * Dynamic query optimization that happens in the middle of your query based on runtime statistics
# MAGIC     * Dynamically coalesce shuffle partitions
# MAGIC     * Dynamically switch join strategies
# MAGIC     * Dynamically optimize skew joins
# MAGIC   * Enable it with: **`spark.sql.adaptive.enabled=true`**
# MAGIC * Dynamic Partition Pruning (DPP)
# MAGIC   * Avoid partition scanning based on the query results of the other query fragments
# MAGIC * Join Hints
# MAGIC * <a href="https://www.youtube.com/watch?v=UZl0pHG-2HA&feature=emb_logo" target="_blank">Improved Pandas UDFs</a>
# MAGIC   * Type Hints
# MAGIC   * Iterators
# MAGIC   * Pandas Function API (mapInPandas, applyInPandas, etc)
# MAGIC * And many more! See the <a href="https://spark.apache.org/docs/latest/api/python/migration_guide/pyspark_2.4_to_3.0.html" target="_blank">migration guide</a> and resources linked above.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
