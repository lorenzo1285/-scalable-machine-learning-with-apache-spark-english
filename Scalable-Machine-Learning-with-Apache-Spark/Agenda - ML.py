# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Agenda
# MAGIC ## Scalable Machine Learning with Apache Spark&trade;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Day 1 AM
# MAGIC | Time | Lesson &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 20m  | **Review**                               | *Review of Spark concepts* |
# MAGIC | 30m    | **ML Overview (optional)**    | Types of Machine Learning, Business applications of ML <br/>(NOTE: this class uses Airbnb's SF rental data to predict things such as price of rental) |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 35m  | **[Data Cleansing]($./ML 01 - Data Cleansing)** | How to deal with null values, outliers, data imputation | 
# MAGIC | 40m  | **[Data Exploration Lab]($./Labs/ML 01L - Data Exploration Lab)**  | Exploring your data, log-normal distribution, determine baseline metric to beat |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 30m    | **[Linear Regression I]($./ML 02 - Linear Regression I)**    | Build simple univariate linear regression model<br/> SparkML APIs: transformer vs estimator |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Day 1 PM
# MAGIC | Time | Lesson &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 20m  | **[Linear Regression I Lab]($./Labs/ML 02L - Linear Regression I Lab)**       | Build multivariate linear regression model <br/> Evaluate RMSE and R2 |
# MAGIC | 30m  | **[Linear Regression II]($./ML 03 - Linear Regression II)**      | How to handle categorical variables (OHE)<br/> Pipeline API <br/>Save and load models|
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 40m |**[Linear Regression II Lab]($./Labs/ML 03L - Linear Regression II Lab)** | Simplify pipeline using RFormula <br/>Build linear regression model to predict on log-scale, then exponentiate prediction and evaluate |
# MAGIC | 30m  | **[MLflow Tracking]($./ML 04 - MLflow Tracking)** | Use MLflow to track experiments, log metrics, and compare runs| 
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 30m  | **[MLflow Model Registry]($./ML 05 - MLflow Model Registry)** | Register a model using MLflow <br/>Deploy that model into production <br/>Update a model in production to new version including a staging phase for testing <br/>Archive and delete models|
# MAGIC | 45m  | **[MLflow Lab]($./Labs/ML 05L - MLflow Lab)** | Use MLflow to track models and Delta table <br/> Register Model|

# COMMAND ----------

# MAGIC %md
# MAGIC ## Day 2 AM
# MAGIC | Time | Lesson &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 20m  | **Review**                               | *Review of Topics* |
# MAGIC | 40m    | **[Decision Trees]($./ML 06 - Decision Trees)**    | Distributed implementation of decision trees and maxBins parameter (why you WILL get different results from sklearn)<br/> Feature importance |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 40m  | **[Random Forests and Hyperparameter Tuning]($./ML 07 - Random Forests and Hyperparameter Tuning)** | Random Forests <br/> K-Fold cross-validation <br/>SparkML's Parallelism parameter (introduced in Spark 2.3) <br/> Speed up Pipeline model training by 4x |                                             ||
# MAGIC | 30m  | **[Hyperparameter Tuning Lab]($./Labs/ML 07L - Hyperparameter Tuning Lab)**  | Perform grid search on a random forest <br/>Get the feature importances across the forest <br/>Save the model <br/>Identify differences between Sklearn's Random Forest and SparkML's |
# MAGIC | 10m  | **Break**   ||
# MAGIC | 20m  | **[Hyperopt]($./ML 08 - Hyperopt)**  | Perform grid search on a random forest <br/>Get the feature importances across the forest <br/>Save the model <br/>Identify differences between Sklearn's Random Forest and SparkML's |
# MAGIC | 20m    | **[Hyperopt Lab]($./Labs/ML 08L - Hyperopt Lab)**    | Distributed hyperparameter tuning for scikit-learn models with SparkTrials |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Day 2 PM
# MAGIC | Time | Lesson &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 25m  | **[AutoML]($./ML 09 - AutoML)**  | Programmatically use Databricks AutoML to automatically train and tune your models |
# MAGIC | 20m    | **[AutoML Lab]($./Labs/ML 09L - AutoML Lab)**    | Use the Databricks AutoML UI to automatically train and tune your models |
# MAGIC | 15m    | **[Feature Store]($./ML 10 - Feature Store)**    | Build, merge, and evolve features with the Databricks Feature Store |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 20m    | **[XGBoost]($./ML 11 - XGBoost)**    | Using 3rd party libraries with Spark <br/>Discuss gradient boosted trees and their variants" |                              
# MAGIC | 15m    | **[Inference with Pandas UDFs]($./ML 12 - Inference with Pandas UDFs)**    | Build a single-node ML model, but apply in parallel using Pandas Scalar Iterator UDF & mapInPandas (introduced in Spark 3.0) |
# MAGIC | 20m    | **[Pandas UDFs Lab]($./Labs/ML 12L - Pandas UDF Lab)**    | Lab to apply a single-node ML model at scale |
# MAGIC | 10m  | **Break** ||
# MAGIC | 15m    | **[Training with Pandas Function API]($./ML 13 - Training with Pandas Function API)**    | Build a single-node ML model for each IoT Device using applyInPandas (introduced in Spark 3.0) and track it with MLflow |
# MAGIC | 20m  | **[Koalas]($./ML 14 - Koalas)** | Use the new open-source library to write Pandas code that distributes using Spark under the hood|  
# MAGIC 
# MAGIC Additional optional notebooks in the electives folder

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
