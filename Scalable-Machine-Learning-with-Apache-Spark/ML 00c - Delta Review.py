# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Delta Review
# MAGIC 
# MAGIC There are a few key operations necessary to understand and make use of <a href="https://docs.delta.io/latest/quick-start.html#create-a-table" target="_blank">Delta Lake</a>.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC - Create a Delta Table
# MAGIC - Read data from your Delta Table
# MAGIC - Update data in your Delta Table
# MAGIC - Access previous versions of your Delta Table using <a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">time travel</a>
# MAGIC - <a href="https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html" target="_blank">Understand the Transaction Log</a>
# MAGIC 
# MAGIC In this notebook we will be using the SF Airbnb rental dataset from <a href="http://insideairbnb.com/get-the-data.html" target="_blank">Inside Airbnb</a>.

# COMMAND ----------

# MAGIC %md ###Why Delta Lake?<br><br>
# MAGIC 
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://user-images.githubusercontent.com/20408077/87175470-4d8e1580-c29e-11ea-8f33-0ee14348a2c1.png" width="500"/>
# MAGIC </div>
# MAGIC 
# MAGIC At a glance, Delta Lake is an open source storage layer that brings both **reliability and performance** to data lakes. Delta Lake provides ACID transactions, scalable metadata handling, and unifies streaming and batch data processing. 
# MAGIC 
# MAGIC Delta Lake runs on top of your existing data lake and is fully compatible with Apache Spark APIs. <a href="https://docs.databricks.com/delta/delta-intro.html" target="_blank">For more information </a>

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md ###Creating a Delta Table
# MAGIC First we need to read the Airbnb dataset as a Spark DataFrame

# COMMAND ----------

file_path = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/"
airbnb_df = spark.read.format("parquet").load(file_path)

display(airbnb_df)

# COMMAND ----------

# MAGIC %md The cell below converts the data to a Delta table using the schema provided by the Spark DataFrame.

# COMMAND ----------

# Converting Spark DataFrame to Delta Table
dbutils.fs.rm(working_dir, True)
airbnb_df.write.format("delta").mode("overwrite").save(working_dir)

# COMMAND ----------

# MAGIC %md A Delta directory can also be registered as a table in the metastore.

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {cleaned_username}")
spark.sql(f"USE {cleaned_username}")

airbnb_df.write.format("delta").mode("overwrite").saveAsTable("delta_review")

# COMMAND ----------

# MAGIC %md Delta supports partitioning. Partitioning puts data with the same value for the partitioned column into its own directory. Operations with a filter on the partitioned column will only read directories that match the filter. This optimization is called partition pruning. Choose partition columns based in the patterns in your data, this dataset for example might benefit if partitioned by neighborhood.

# COMMAND ----------

airbnb_df.write.format("delta").mode("overwrite").partitionBy("neighbourhood_cleansed").option("overwriteSchema", "true").save(working_dir)

# COMMAND ----------

# MAGIC %md ###Understanding the <a href="https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html" target="_blank">Transaction Log </a>
# MAGIC Let's take a look at the Delta Transaction Log. We can see how Delta stores the different neighborhood partitions in separate files. Additionally, we can also see a directory called _delta_log.

# COMMAND ----------

display(dbutils.fs.ls(working_dir))

# COMMAND ----------

# MAGIC %md <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://user-images.githubusercontent.com/20408077/87174138-609fe600-c29c-11ea-90cc-84df0c1357f1.png" width="500"/>
# MAGIC </div>
# MAGIC 
# MAGIC When a user creates a Delta Lake table, that table’s transaction log is automatically created in the _delta_log subdirectory. As he or she makes changes to that table, those changes are recorded as ordered, atomic commits in the transaction log. Each commit is written out as a JSON file, starting with 000000.json. Additional changes to the table generate more JSON files.

# COMMAND ----------

display(dbutils.fs.ls(working_dir + "/_delta_log/"))

# COMMAND ----------

# MAGIC %md Next, let's take a look at a Transaction Log File.
# MAGIC 
# MAGIC The <a href="https://docs.databricks.com/delta/delta-utility.html" target="_blank">four columns</a> each represent a different part of the very first commit to the Delta Table where the table was created.<br><br>
# MAGIC 
# MAGIC - The add column has statistics about the DataFrame as a whole and individual columns.
# MAGIC - The commitInfo column has useful information about what the operation was (WRITE or READ) and who executed the operation.
# MAGIC - The metaData column contains information about the column schema.
# MAGIC - The protocol version contains information about the minimum Delta version necessary to either write or read to this Delta Table.

# COMMAND ----------

display(spark.read.json(working_dir + "/_delta_log/00000000000000000000.json"))

# COMMAND ----------

# MAGIC %md The second transaction log has 39 rows. This includes metadata for each partition. 

# COMMAND ----------

display(spark.read.json(working_dir + "/_delta_log/00000000000000000001.json"))

# COMMAND ----------

# MAGIC %md Finally, let's take a look at the files inside one of the Neighborhood partitions. The file inside corresponds to the partition commit (file 01) in the _delta_log directory.

# COMMAND ----------

display(dbutils.fs.ls(working_dir + "/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md ### Reading data from your Delta table

# COMMAND ----------

df = spark.read.format("delta").load(working_dir)
display(df)

# COMMAND ----------

# MAGIC %md #Updating your Delta Table
# MAGIC 
# MAGIC Let's filter for rows where the host is a superhost.

# COMMAND ----------

df_update = airbnb_df.filter(airbnb_df["host_is_superhost"] == "t")
display(df_update)

# COMMAND ----------

df_update.write.format("delta").mode("overwrite").save(working_dir)

# COMMAND ----------

df = spark.read.format("delta").load(working_dir)
display(df)

# COMMAND ----------

# MAGIC %md Let's look at the files in the Bayview partition post-update. Remember, the different files in this directory are snapshots of your DataFrame corresponding to different commits.

# COMMAND ----------

display(dbutils.fs.ls(working_dir + "/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md #Delta Time Travel

# COMMAND ----------

# MAGIC %md Oops, actually we need the entire dataset! You can access a previous version of your Delta Table using <a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">Delta Time Travel</a>. Use the following two cells to access your version history. Delta Lake will keep a 30 day version history by default, though it can maintain that history for longer if needed.

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS train_delta")
spark.sql(f"CREATE TABLE train_delta USING DELTA LOCATION '{working_dir}'")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY train_delta

# COMMAND ----------

# MAGIC %md Using the **`versionAsOf`** option allows you to easily access previous versions of our Delta Table.

# COMMAND ----------

df = spark.read.format("delta").option("versionAsOf", 0).load(working_dir)
display(df)

# COMMAND ----------

# MAGIC %md You can also access older versions using a timestamp.
# MAGIC 
# MAGIC Replace the timestamp string with the information from your version history. Note that you can use a date without the time information if necessary.

# COMMAND ----------

# Use your own timestamp 
# time_stamp_string = "FILL_IN"

# OR programatically get the first verion's timestamp value
time_stamp_string = str(spark.sql("DESCRIBE HISTORY train_delta").collect()[-1]["timestamp"])

df = spark.read.format("delta").option("timestampAsOf", time_stamp_string).load(working_dir)
display(df)

# COMMAND ----------

# MAGIC %md Now that we're happy with our Delta Table, we can clean up our directory using **`VACUUM`**. Vacuum accepts a retention period in hours as an input.

# COMMAND ----------

# MAGIC %md Uh-oh, our code doesn't run! By default, to prevent accidentally vacuuming recent commits, Delta Lake will not let users vacuum a period under 7 days or 168 hours. Once vacuumed, you cannot return to a prior commit through time travel, only your most recent Delta Table will be saved.
# MAGIC 
# MAGIC Try changing the vacuum parameter to different values.

# COMMAND ----------

# from delta.tables import DeltaTable

# delta_table = DeltaTable.forPath(spark, working_dir)
# delta_table.vacuum(0)

# COMMAND ----------

# MAGIC %md We can workaround this by setting a spark configuration that will bypass the default retention period check.

# COMMAND ----------

from delta.tables import DeltaTable

spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
delta_table = DeltaTable.forPath(spark, working_dir)
delta_table.vacuum(0)

# COMMAND ----------

# MAGIC %md Let's take a look at our Delta Table files now. After vacuuming, the directory only holds the partition of our most recent Delta Table commit.

# COMMAND ----------

display(dbutils.fs.ls(working_dir + "/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md Since vacuuming deletes files referenced by the Delta Table, we can no longer access past versions. The code below should throw an error.

# COMMAND ----------

# df = spark.read.format("delta").option("versionAsOf", 0).load(working_dir)
# display(df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
