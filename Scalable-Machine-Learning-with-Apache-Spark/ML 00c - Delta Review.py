# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Delta Review
# MAGIC 
# MAGIC There are a few key operations necessary to understand and make use of [Delta Lake](https://docs.delta.io/latest/quick-start.html#create-a-table).
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC - Create a Delta Table
# MAGIC - Read data from your Delta Table
# MAGIC - Update data in your Delta Table
# MAGIC - Access previous versions of your Delta Table using [time travel](https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html) 
# MAGIC - [Understand the Transaction Log](https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html)
# MAGIC 
# MAGIC In this notebook we will be using the SF Airbnb rental dataset from [Inside Airbnb](http://insideairbnb.com/get-the-data.html).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ###Why Delta Lake?<br><br>
# MAGIC 
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://user-images.githubusercontent.com/20408077/87175470-4d8e1580-c29e-11ea-8f33-0ee14348a2c1.png" width="500"/>
# MAGIC </div>
# MAGIC 
# MAGIC At a glance, Delta Lake is an open source storage layer that brings both **reliability and performance** to data lakes. Delta Lake provides ACID transactions, scalable metadata handling, and unifies streaming and batch data processing. 
# MAGIC 
# MAGIC Delta Lake runs on top of your existing data lake and is fully compatible with Apache Spark APIs. [For more information](https://docs.databricks.com/delta/delta-intro.html)

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ###Creating a Delta Table
# MAGIC First we need to read the Airbnb dataset as a Spark DataFrame

# COMMAND ----------

filePath = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/"
airbnbDF = spark.read.parquet(filePath)

display(airbnbDF)

# COMMAND ----------

# MAGIC %md
# MAGIC The cell below converts the data to a Delta table using the schema provided by the Spark DataFrame.

# COMMAND ----------

# Converting Spark DataFrame to Delta Table
deltaPath = userhome + "/delta-p"
dbutils.fs.rm(deltaPath, True)
airbnbDF.write.format("delta").mode("overwrite").save(deltaPath)

# COMMAND ----------

# MAGIC %md
# MAGIC You can also create a Delta table in the metastore.

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {cleaned_username}")
spark.sql(f"USE {cleaned_username}")

airbnbDF.write.format("delta").mode("overwrite").saveAsTable("deltaReview")

# COMMAND ----------

# MAGIC %md
# MAGIC Delta supports partitioning your data using unique values in a specified column. Let's partition by the neighborhood column. Partitioning by neighborhood gives us a point of quick comparison between different parts of San Francisco.

# COMMAND ----------

airbnbDF.write.format("delta").mode("overwrite").partitionBy("neighbourhood_cleansed").option("overwriteSchema", "true").save(deltaPath)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Understanding the [Transaction Log](https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html)
# MAGIC Let's take a look at the Delta Transaction Log. We can see how Delta stores the different neighborhood partitions in separate files. Additionally, we can also see a directory called _delta_log.

# COMMAND ----------

display(dbutils.fs.ls(deltaPath))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://user-images.githubusercontent.com/20408077/87174138-609fe600-c29c-11ea-90cc-84df0c1357f1.png" width="500"/>
# MAGIC </div>
# MAGIC 
# MAGIC When a user creates a Delta Lake table, that table’s transaction log is automatically created in the _delta_log subdirectory. As he or she makes changes to that table, those changes are recorded as ordered, atomic commits in the transaction log. Each commit is written out as a JSON file, starting with 000000.json. Additional changes to the table generate subsequent JSON files in ascending numerical order so that the next commit is written out as 000001.json, the following as 000002.json, and so on.

# COMMAND ----------

display(dbutils.fs.ls(deltaPath + "/_delta_log/"))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's take a look at a Transaction Log File.
# MAGIC 
# MAGIC The [four columns](https://docs.databricks.com/delta/delta-utility.html) each represent a different part of the very first commit to the Delta Table, creating the table.
# MAGIC - The add column has statistics about the DataFrame as a whole and individual columns.
# MAGIC - The commitInfo column has useful information about what the operation was (WRITE or READ) and who executed the operation.
# MAGIC - The metaData column contains information about the column schema.
# MAGIC - The protocol version contains information about the minimum Delta version necessary to either write or read to this Delta Table.

# COMMAND ----------

display(spark.read.json(deltaPath + "/_delta_log/00000000000000000000.json"))

# COMMAND ----------

# MAGIC %md
# MAGIC One key difference between these two transaction logs is the size of the JSON file, this file has 39 rows compared to the previous 4. To understand why, let's take a look at the commitInfo column. We can see that in the operationParameters section, partitionBy has been filled in by the "neighbourhood_cleansed" column. Furthermore, if we look at the add section on row 3, we can see that a new section called partitionValues has appeared. As we saw above, Delta stores partitions separately in memory, however, it stores information about these partitions in the same transaction log file.

# COMMAND ----------

display(spark.read.json(deltaPath + "/_delta_log/00000000000000000001.json"))

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, let's take a look at the files inside one of the Neighborhood partitions. The file inside corresponds to the partition commit (file 01) in the _delta_log directory.

# COMMAND ----------

display(dbutils.fs.ls(deltaPath + "/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading data from your Delta table

# COMMAND ----------

df = spark.read.format("delta").load(deltaPath)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Updating your Delta Table
# MAGIC 
# MAGIC Let's filter for rows where the host is a superhost.

# COMMAND ----------

df_update = airbnbDF.filter(airbnbDF["host_is_superhost"] == "t")
display(df_update)

# COMMAND ----------

df_update.write.format("delta").mode("overwrite").save(deltaPath)

# COMMAND ----------

df = spark.read.format("delta").load(deltaPath)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look at the files in the Bayview partition post-update. Remember, the different files in this directory are snapshots of your DataFrame corresponding to different commits.

# COMMAND ----------

display(dbutils.fs.ls(deltaPath + "/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md
# MAGIC #Delta Time Travel

# COMMAND ----------

# MAGIC %md
# MAGIC Oops, actually we need the entire dataset! You can access a previous version of your Delta Table using [Delta Time Travel](https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html). Use the following two cells to access your version history. Delta Lake will keep a 30 day version history by default, but if necessary, Delta can store a version history for longer.

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS train_delta")
spark.sql(f"CREATE TABLE train_delta USING DELTA LOCATION '{deltaPath}'")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY train_delta

# COMMAND ----------

# MAGIC %md
# MAGIC Using the `versionAsOf` option allows you to easily access previous versions of our Delta Table.

# COMMAND ----------

df = spark.read.format("delta").option("versionAsOf", 0).load(deltaPath)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC You can also access older versions using a timestamp.
# MAGIC 
# MAGIC Replace the timestamp string with the information from your version history. Note that you can use a date without the time information if necessary.

# COMMAND ----------

# TODO
timeStampString = <FILL_IN>
df = spark.read.format("delta").option("timestampAsOf", timeStampString).load(deltaPath)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we're happy with our Delta Table, we can clean up our directory using `VACUUM`. Vacuum accepts a retention period in hours as an input.

# COMMAND ----------

# MAGIC %md
# MAGIC Uh-oh, our code doesn't run! By default, to prevent accidentally vacuuming recent commits, Delta Lake will not let users vacuum a period under 7 days or 168 hours. Once vacuumed, you cannot return to a prior commit through time travel, only your most recent Delta Table will be saved.

# COMMAND ----------

# from delta.tables import *

# deltaTable = DeltaTable.forPath(spark, deltaPath)
# deltaTable.vacuum(0)

# COMMAND ----------

# MAGIC %md
# MAGIC We can workaround this by setting a spark configuration that will bypass the default retention period check.

# COMMAND ----------

from delta.tables import *

spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
deltaTable = DeltaTable.forPath(spark, deltaPath)
deltaTable.vacuum(0)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at our Delta Table files now. After vacuuming, the directory only holds the partition of our most recent Delta Table commit.

# COMMAND ----------

display(dbutils.fs.ls(deltaPath + "/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md
# MAGIC Since vacuuming deletes files referenced by the Delta Table, we can no longer access past versions. The code below should throw an error.

# COMMAND ----------

# df = spark.read.format("delta").option("versionAsOf", 0).load(deltaPath)
# display(df)


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
