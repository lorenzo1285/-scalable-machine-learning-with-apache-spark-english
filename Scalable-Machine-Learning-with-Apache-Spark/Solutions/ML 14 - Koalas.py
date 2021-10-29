# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://raw.githubusercontent.com/databricks/koalas/master/Koalas-logo.png" width="220"/>
# MAGIC </div>
# MAGIC 
# MAGIC The Koalas project makes data scientists more productive when interacting with big data, by implementing the pandas DataFrame API on top of Apache Spark. By unifying the two ecosystems with a familiar API, Koalas offers a seamless transition between small and large data.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC - Demonstrate the similarities of the Koalas API with the pandas API
# MAGIC - Understand the differences in syntax for the same DataFrame operations in Koalas vs PySpark
# MAGIC 
# MAGIC [Koalas Docs](https://koalas.readthedocs.io/en/latest/index.html), [Koalas Github](https://github.com/databricks/koalas), Spark+AI Summit Talks from [Niall Turbitt](https://www.youtube.com/watch?v=iUpBSHoqzLM&feature=youtu.be) & [Takuya Ueshin](https://www.youtube.com/watch?v=G_-9VbyHcx8&feature=youtu.be)
# MAGIC 
# MAGIC `koalas` comes pre-installed on the Machine Learning Runtime.

# COMMAND ----------

# MAGIC %md ### [Performance](https://databricks.com/blog/2019/08/22/guest-blog-how-virgin-hyperloop-one-reduced-processing-time-from-hours-to-minutes-with-koalas.html)
# MAGIC 
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2019/08/koalas-image4.png" width="1000"/>
# MAGIC </div>
# MAGIC 
# MAGIC **Pandas** DataFrames are mutable, eagerily evaluated, and maintain row order. They are restricted to a single machine, and are very performant when the data sets are small, as shown in a).
# MAGIC 
# MAGIC **Spark** DataFrames are distributed, lazily evaluated, immutable, and do not maintain row order. They are very performant when working at scale, as shown in b) and c).
# MAGIC 
# MAGIC **Koalas** provides the best of both worlds: pandas API with the performance benefits of Spark. However, it is not as fast as implementing your solution natively in Spark, and let's see why below.

# COMMAND ----------

# MAGIC %md ## InternalFrame
# MAGIC 
# MAGIC The InternalFrame holds the current Spark DataFrame and internal immutable metadata.
# MAGIC 
# MAGIC It manages mappings from Koalas column names to Spark column names, as well as from Koalas index names to Spark column names. 
# MAGIC 
# MAGIC If a user calls some API, the Koalas DataFrame updates the Spark DataFrame and metadata in InternalFrame. It creates or copies the current InternalFrame with the new states, and returns a new Koalas DataFrame.
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/301/InternalFrame.png)

# COMMAND ----------

# MAGIC %md ## InternalFrame Metadata Updates Only
# MAGIC 
# MAGIC Sometimes the update of Spark DataFrame is not needed but of metadata only, then new structure will be like this.
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/301/InternalFrameMetadata.png)

# COMMAND ----------

# MAGIC %md ## InternalFrame Inplace Updates
# MAGIC 
# MAGIC On the other hand, sometimes Koalas DataFrame updates internal state instead of returning a new DataFrame, for example, the argument  inplace=True is provided, then new structure will be like this.
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/301/InternalFrameInPlace.png)

# COMMAND ----------

# MAGIC %md ### Read in the dataset
# MAGIC 
# MAGIC * PySpark
# MAGIC * pandas
# MAGIC * Koalas

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md Read in Parquet with PySpark

# COMMAND ----------

df = spark.read.parquet(f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
display(df)

# COMMAND ----------

# MAGIC %md Read in Parquet with pandas

# COMMAND ----------

import pandas as pd

pdf = pd.read_parquet(f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/".replace("dbfs:/", "/dbfs/"))
pdf.head()

# COMMAND ----------

# MAGIC %md Read in Parquet with Koalas. You'll notice Koalas generates an index column for you, like in pandas.
# MAGIC 
# MAGIC Koalas also supports reading from Delta (`read_delta`), but pandas does not support that yet.

# COMMAND ----------

import databricks.koalas as ks

kdf = ks.read_parquet(f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
kdf.head()

# COMMAND ----------

# MAGIC %md ### [Index Types](https://koalas.readthedocs.io/en/latest/user_guide/options.html#default-index-type)
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/301/koalas_index.png)

# COMMAND ----------

ks.set_option("compute.default_index_type", "distributed-sequence")
kdf_dist_sequence = ks.read_parquet(f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
kdf_dist_sequence.head()

# COMMAND ----------

# MAGIC %md ### Converting to Koalas DataFrame to/from Spark DataFrame

# COMMAND ----------

# MAGIC %md Creating a Koalas DataFrame from PySpark DataFrame

# COMMAND ----------

kdf = ks.DataFrame(df)
display(kdf)

# COMMAND ----------

# MAGIC %md Alternative way of creating a Koalas DataFrame from PySpark DataFrame

# COMMAND ----------

kdf = df.to_koalas()
display(kdf)

# COMMAND ----------

# MAGIC %md Go from a Koalas DataFrame to a Spark DataFrame

# COMMAND ----------

display(kdf.to_spark())

# COMMAND ----------

# MAGIC %md ### Value Counts

# COMMAND ----------

# MAGIC %md Get value counts of the different property types with PySpark

# COMMAND ----------

display(df.groupby("property_type").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md Get value counts of the different property types with Koalas

# COMMAND ----------

kdf["property_type"].value_counts()

# COMMAND ----------

# MAGIC %md ### Visualizations with Koalas DataFrames

# COMMAND ----------

ks.options.plotting.backend='matplotlib'
kdf[["bedrooms", "price"]].plot.hist(x="bedrooms", y="price", bins=200)

# COMMAND ----------

graph_kdf = kdf.filter(items=["bedrooms", "price"])
graph_kdf.plot.hist(x="bedrooms", y="price", bins=200)

# COMMAND ----------

# MAGIC %md ### SQL on Koalas DataFrames

# COMMAND ----------

ks.sql("select distinct(property_type) from {kdf}")

# COMMAND ----------

# MAGIC %md ### Interesting Facts
# MAGIC 
# MAGIC * With Koalas you can read from Delta Tables and read in a directory of files
# MAGIC * If you use apply on a Koalas DF and that DF is <1000 (by default), Koalas will use pandas as a shortcut - this can be adjusted using `compute.shortcut_limit`
# MAGIC * When you create bar plots, the top n rows are only used - this can be adjusted using `plotting.max_rows`
# MAGIC * How to utilize `.apply` ([docs](https://koalas.readthedocs.io/en/latest/reference/api/databricks.koalas.DataFrame.apply.html#databricks.koalas.DataFrame.apply)) with its use of return type hints similar to pandas UDFs
# MAGIC * How to check the execution plan, as well as caching a Koalas DF (which aren't immediately intuitive)
# MAGIC * Koalas are marsupials whose max speed is 30 kph (20 mph)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
