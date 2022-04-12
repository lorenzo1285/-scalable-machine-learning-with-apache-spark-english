# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://files.training.databricks.com/images/301/deployment_options_mllib.png)
# MAGIC 
# MAGIC There are four main deployment options:
# MAGIC * Batch pre-compute
# MAGIC * Structured streaming
# MAGIC * Low-latency model serving
# MAGIC * Mobile/embedded (outside scope of class)
# MAGIC 
# MAGIC We have already seen how to do batch predictions using Spark. Now let's look at how to make predictions on streaming data.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Apply a SparkML model on a simulated stream of data 

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md ## Load in Model & Data
# MAGIC 
# MAGIC We are loading in a repartitioned version of our dataset (100 partitions instead of 4) to see more incremental progress of the streaming predictions.

# COMMAND ----------

from pyspark.ml.pipeline import PipelineModel

pipeline_path = f"{datasets_dir}/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model"
pipeline_model = PipelineModel.load(pipeline_path)

repartitioned_path =  f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/"
schema = spark.read.parquet(repartitioned_path).schema

# COMMAND ----------

# MAGIC %md ## Simulate streaming data
# MAGIC 
# MAGIC **NOTE**: You must specify a schema when creating a streaming source DataFrame.

# COMMAND ----------

streaming_data = (spark
                 .readStream
                 .schema(schema) # Can set the schema this way
                 .option("maxFilesPerTrigger", 1)
                 .parquet(repartitioned_path))

# COMMAND ----------

# MAGIC %md ## Make Predictions

# COMMAND ----------

stream_pred = pipeline_model.transform(streaming_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's save our results.

# COMMAND ----------

import re

checkpoint_dir = working_dir + "/stream_checkpoint"
# Clear out the checkpointing directory
dbutils.fs.rm(checkpoint_dir, True) 

(stream_pred
 .writeStream
 .format("memory")
 .option("checkpointLocation", checkpoint_dir)
 .outputMode("append")
 .queryName("pred_stream")
 .start())

# COMMAND ----------

untilStreamIsReady("pred_stream")

# COMMAND ----------

# MAGIC %md While this is running, take a look at the new Structured Streaming tab in the Spark UI.

# COMMAND ----------

display(
  sql("select * from pred_stream")
)

# COMMAND ----------

display(
  sql("select count(*) from pred_stream")
)

# COMMAND ----------

# MAGIC %md Now that we are done, make sure to stop the stream

# COMMAND ----------

for stream in spark.streams.active:
    print(f"Stopping {stream.name}")
    stream.stop()                  # Stop the active stream
    try: stream.awaitTermination() # Wait for it to actually stop
    except: pass                   # Don't care if stopping fails

# COMMAND ----------

# MAGIC %md ### What about Model Export?
# MAGIC 
# MAGIC * <a href="https://onnx.ai/" target="_blank">ONNX</a>
# MAGIC   * ONNX is very popular in the deep learning community allowing developers to switch between libraries and languages, but only has experimental support for MLlib.
# MAGIC * DIY (Reimplement it yourself)
# MAGIC   * Error-prone, fragile
# MAGIC * 3rd party libraries
# MAGIC   * See XGBoost notebook
# MAGIC   * <a href="https://www.h2o.ai/products/h2o-sparkling-water/" target="_blank">H2O</a>

# COMMAND ----------

# MAGIC %md ### Low-Latency Serving Solutions
# MAGIC 
# MAGIC Low-latency serving can operate as quickly as tens to hundreds of milliseconds.  Custom solutions are normally backed by Docker and/or Flask (though Flask generally isn't recommended in production unless significant precations are taken).  Managed solutions also include:<br><br>
# MAGIC 
# MAGIC * <a href="https://docs.databricks.com/applications/mlflow/model-serving.html" target="_blank">MLflow Model Serving</a>
# MAGIC * <a href="https://azure.microsoft.com/en-us/services/machine-learning/" target="_blank">Azure Machine Learning</a>
# MAGIC * <a href="https://aws.amazon.com/sagemaker/" target="_blank">SageMaker</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
