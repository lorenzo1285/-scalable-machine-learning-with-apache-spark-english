# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Training with Pandas Function API
# MAGIC 
# MAGIC This notebook demonstrates how to use Pandas Function API to manage and scale machine learning models for IoT devices. 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.GroupedData.applyInPandas.html?highlight=applyinpandas#pyspark.sql.GroupedData.applyInPandas" target="_blank"> **`.groupBy().applyInPandas()`** </a> to build many models in parallel for each IoT Device

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md Create dummy data with:
# MAGIC - **`device_id`**: 10 different devices
# MAGIC - **`record_id`**: 10k unique records
# MAGIC - **`feature_1`**: a feature for model training
# MAGIC - **`feature_2`**: a feature for model training
# MAGIC - **`feature_3`**: a feature for model training
# MAGIC - **`label`**: the variable we're trying to predict

# COMMAND ----------

import pyspark.sql.functions as f

df = (spark
      .range(1000*100)
      .select(f.col("id").alias("record_id"), (f.col("id")%10).alias("device_id"))
      .withColumn("feature_1", f.rand() * 1)
      .withColumn("feature_2", f.rand() * 2)
      .withColumn("feature_3", f.rand() * 3)
      .withColumn("label", (f.col("feature_1") + f.col("feature_2") + f.col("feature_3")) + f.rand())
     )

display(df)

# COMMAND ----------

# MAGIC %md Define the return schema

# COMMAND ----------

import pyspark.sql.types as t

train_return_schema = t.StructType([
    t.StructField("device_id", t.IntegerType()), # unique device ID
    t.StructField("n_used", t.IntegerType()),    # number of records used in training
    t.StructField("model_path", t.StringType()), # path to the model for a given device
    t.StructField("mse", t.FloatType())          # metric for model performance
])

# COMMAND ----------

# MAGIC %md Define a pandas function that takes all the data for a given device, train a model, saves it as a nested run, and returns a spark object with the above schema

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Trains an sklearn model on grouped instances
    """
    # Pull metadata
    device_id = df_pandas["device_id"].iloc[0]
    n_used = df_pandas.shape[0]
    run_id = df_pandas["run_id"].iloc[0] # Pulls run ID to do a nested run

    # Train the model
    X = df_pandas[["feature_1", "feature_2", "feature_3"]]
    y = df_pandas["label"]
    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Evaluate the model
    predictions = rf.predict(X)
    mse = mean_squared_error(y, predictions) # Note we could add a train/test split

    # Resume the top-level training
    with mlflow.start_run(run_id=run_id) as outer_run:
        # Small hack for for running as a job
        experiment_id = outer_run.info.experiment_id
        print(f"Current experiment_id = {experiment_id}")

        # Create a nested run for the specific device
        with mlflow.start_run(run_name=str(device_id), nested=True, experiment_id=experiment_id) as run:
            mlflow.sklearn.log_model(rf, str(device_id))
            mlflow.log_metric("mse", mse)

            artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
            # Create a return pandas DataFrame that matches the schema above
            return_df = pd.DataFrame([[device_id, n_used, artifact_uri, mse]], 
                                    columns=["device_id", "n_used", "model_path", "mse"])

    return return_df 


# COMMAND ----------

# MAGIC %md Apply the pandas function to grouped data. 
# MAGIC 
# MAGIC Note that the way you would apply this in practice depends largely on where the data for inference is located. In this example, we'll reuse the training data which contains our device and run id's.

# COMMAND ----------

with mlflow.start_run(run_name="Training session for all devices") as run:
    run_id = run.info.run_id

    model_directories_df = (df
        .withColumn("run_id", f.lit(run_id)) # Add run_id
        .groupby("device_id")
        .applyInPandas(train_model, schema=train_return_schema)
        .cache()
    )

combined_df = df.join(model_directories_df, on="device_id", how="left")
display(combined_df)

# COMMAND ----------

# MAGIC %md Define a pandas function to apply the model.  *This needs only one read from DBFS per device.*

# COMMAND ----------

apply_return_schema = t.StructType([
    t.StructField("record_id", t.IntegerType()),
    t.StructField("prediction", t.FloatType())
])

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Applies model to data for a particular device, represented as a pandas DataFrame
    """
    model_path = df_pandas["model_path"].iloc[0]

    input_columns = ["feature_1", "feature_2", "feature_3"]
    X = df_pandas[input_columns]

    model = mlflow.sklearn.load_model(model_path)
    prediction = model.predict(X)

    return_df = pd.DataFrame({
        "record_id": df_pandas["record_id"],
        "prediction": prediction
    })
    return return_df

prediction_df = combined_df.groupby("device_id").applyInPandas(apply_model, schema=apply_return_schema)
display(prediction_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
