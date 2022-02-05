# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Databricks Best Practices
# MAGIC 
# MAGIC In this notebook, we will explore a wide array of best practices for working with Databricks.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Explore a general framework for debugging slow running jobs
# MAGIC  - Identify the security implications of various data access paradigms
# MAGIC  - Determine various cluster configuration issues including machine types, libraries, and jobs
# MAGIC  - Integrate Databricks notebooks and jobs with version control and the CLI

# COMMAND ----------

# MAGIC %md ## Slow Running Jobs
# MAGIC 
# MAGIC The most common issues with slow running jobs are:<br><br>
# MAGIC 
# MAGIC - `Spill`: Data is exhausting the cluster's memory and is spilling onto disk. Resolution: a cluster with more memory resources
# MAGIC - `Shuffle`: Large amounts of data are being transferred across the cluster.  Resolution: optimize joins or refactor code to avoid shuffles
# MAGIC - `Skew/Stragglers`: Partitioned data (in files or in memory) is skewed causing the "curse of the last reducer" where some partitions take longer to run.  Resolution: repartition to a multiple of the available cores or use skew hints
# MAGIC - `Small/Large Files`: Too many small files are exhausting cluster resources since each file read needs its own thread or few large files are causing unused threads.  Resolution: rewrite data in a more optimized way or perform Delta file compaction
# MAGIC 
# MAGIC Your debugging toolkit:<br><br>
# MAGIC 
# MAGIC - Ganglia for CPU, network, and memory resources at a cluster or node level
# MAGIC - Spark UI for most everything else (especially the storage and executor tabs)
# MAGIC - Driver or worker logs for errors (especially with background processes)
# MAGIC - Notebook tab of the clusters section to see if the intern is hogging your cluster again

# COMMAND ----------

# MAGIC %md ## Data Access and Security
# MAGIC 
# MAGIC A few notes on data access:<br><br>
# MAGIC 
# MAGIC * [Mount data for easy access](https://docs.databricks.com/data/databricks-file-system.html#mount-storage)
# MAGIC * [Use secrets to secure credentials](https://docs.databricks.com/dev-tools/cli/secrets-cli.html#secrets-cli) (this keeps credentials out of the code)
# MAGIC * Credential passthrough works in [AWS](https://docs.databricks.com/dev-tools/cli/secrets-cli.html#secrets-cli) and [Azure](https://docs.microsoft.com/en-us/azure/databricks/security/credential-passthrough/adls-passthrough)

# COMMAND ----------

# MAGIC %md ## Cluster Configuration, Libraries, and Jobs
# MAGIC 
# MAGIC Cluster types are:<br><br>
# MAGIC 
# MAGIC - Memory optimized (with or without [Delta Cache Acceleration](https://docs.databricks.com/delta/optimizations/delta-cache.html))
# MAGIC - Compute optimized
# MAGIC - Storage optimized
# MAGIC - GPU accelerated
# MAGIC - General Purpose
# MAGIC 
# MAGIC General rules of thumb:<br><br>
# MAGIC 
# MAGIC - Smaller clusters of larger machine types for machine learning
# MAGIC - One cluster per production workload
# MAGIC - Don't share clusters for ML training (even in development)
# MAGIC - [See the docs for more specifics](https://docs.databricks.com/clusters/configure.html)

# COMMAND ----------

# MAGIC %md Library installation best practices:<br><br>
# MAGIC   
# MAGIC - [Notebook-scoped Python libraries](https://docs.databricks.com/libraries/notebooks-python-libraries.html) ensure users on same cluster can have different libraries.  Also good for saving notebooks with their library dependencies
# MAGIC - [Init scripts](https://docs.databricks.com/clusters/init-scripts.html) ensure that code is ran before the JVM starts (good for certain libraries or environment configuration)
# MAGIC - Some configuration variables need to be set on cluster start

# COMMAND ----------

# MAGIC %md Jobs best practices:<br><br>
# MAGIC 
# MAGIC - Use [notebook workflows](https://docs.databricks.com/notebooks/notebook-workflows.html)
# MAGIC - [Widgets](https://docs.databricks.com/notebooks/widgets.html) work for parameter passing
# MAGIC - You can also run jars and wheels
# MAGIC - Use the CLI for orchestration tools (e.g. Airflow)
# MAGIC - [See the docs for more specifics](https://docs.databricks.com/jobs.html)
# MAGIC - Always specify a timeout interval to prevent infinitely running jobs

# COMMAND ----------

# MAGIC %md ## CLI and Version Control
# MAGIC 
# MAGIC The [Databricks CLI](https://github.com/databricks/databricks-cli):<br><br>
# MAGIC 
# MAGIC  * Programmatically export out all your notebooks to check into github
# MAGIC  * Can also import/export data, execute jobs, create clusters, and perform most other Workspace tasks
# MAGIC 
# MAGIC Git integration can be accomplished in a few ways:<br><br>
# MAGIC 
# MAGIC  * Use the CLI to import/export notebooks and check into git manually
# MAGIC  * [Use the built-in git integration](https://docs.databricks.com/notebooks/github-version-control.html)
# MAGIC  * [Use the next generation workspace for alternative project integration](https://www.youtube.com/watch?v=HsfMmBfQtvI)

# COMMAND ----------

# MAGIC %md Time permitting: exploring the [admin console!](https://docs.databricks.com/administration-guide/index.html)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
