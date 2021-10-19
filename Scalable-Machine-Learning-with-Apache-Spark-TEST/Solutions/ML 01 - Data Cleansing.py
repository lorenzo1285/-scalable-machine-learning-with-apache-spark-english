# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Cleansing with Airbnb
# MAGIC 
# MAGIC We're going to start by doing some exploratory data analysis & cleansing. We will be using the SF Airbnb rental dataset from [Inside Airbnb](http://insideairbnb.com/get-the-data.html).
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/301/sf.jpg" style="height: 200px; margin: 10px; border: 1px solid #ddd; padding: 10px"/>
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Impute missing values
# MAGIC  - Identify & remove outliers

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC Let's load the Airbnb dataset in.

# COMMAND ----------

filePath = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06.csv"

rawDF = spark.read.csv(filePath, header="true", inferSchema="true", multiLine="true", escape='"')

display(rawDF)

# COMMAND ----------

rawDF.columns

# COMMAND ----------

# MAGIC %md
# MAGIC For the sake of simplicity, only keep certain columns from this dataset. We will talk about feature selection later.

# COMMAND ----------

columnsToKeep = [
  "host_is_superhost",
  "cancellation_policy",
  "instant_bookable",
  "host_total_listings_count",
  "neighbourhood_cleansed",
  "latitude",
  "longitude",
  "property_type",
  "room_type",
  "accommodates",
  "bathrooms",
  "bedrooms",
  "beds",
  "bed_type",
  "minimum_nights",
  "number_of_reviews",
  "review_scores_rating",
  "review_scores_accuracy",
  "review_scores_cleanliness",
  "review_scores_checkin",
  "review_scores_communication",
  "review_scores_location",
  "review_scores_value",
  "price"]

baseDF = rawDF.select(columnsToKeep)
baseDF.cache().count()
display(baseDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Fixing Data Types
# MAGIC 
# MAGIC Take a look at the schema above. You'll notice that the `price` field got picked up as string. For our task, we need it to be a numeric (double type) field. 
# MAGIC 
# MAGIC Let's fix that.

# COMMAND ----------

from pyspark.sql.functions import col, translate

fixedPriceDF = baseDF.withColumn("price", translate(col("price"), "$,", "").cast("double"))

display(fixedPriceDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary statistics
# MAGIC 
# MAGIC Two options:
# MAGIC * describe
# MAGIC * summary (describe + IQR)
# MAGIC 
# MAGIC **Question:** When to use IQR/median over mean? Vice versa?

# COMMAND ----------

# MAGIC %md 
# MAGIC // INSTRUCTOR_NOTES
# MAGIC 
# MAGIC The mean is cheaper to compute than the median, and if the data is normally distributed, then the two contain the same value. However, with outliers, you should be looking at the median. Given that you don't know a priori if there are outliers or not, you should use the median/IQR.

# COMMAND ----------

display(fixedPriceDF.describe())

# COMMAND ----------

display(fixedPriceDF.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Getting rid of extreme values
# MAGIC 
# MAGIC Let's take a look at the *min* and *max* values of the `price` column:

# COMMAND ----------

display(fixedPriceDF.select("price").describe())

# COMMAND ----------

# MAGIC %md There are some super-expensive listings. But it's the data scientist's job to decide what to do with them. We can certainly filter the "free" Airbnbs though.
# MAGIC 
# MAGIC Let's see first how many listings we can find where the *price* is zero.

# COMMAND ----------

fixedPriceDF.filter(col("price") == 0).count()

# COMMAND ----------

# MAGIC %md
# MAGIC Now only keep rows with a strictly positive *price*.

# COMMAND ----------

posPricesDF = fixedPriceDF.filter(col("price") > 0)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the *min* and *max* values of the *minimum_nights* column:

# COMMAND ----------

display(posPricesDF.select("minimum_nights").describe())

# COMMAND ----------

display(posPricesDF
  .groupBy("minimum_nights").count()
  .orderBy(col("count").desc(), col("minimum_nights"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC A minimum stay of one year seems to be a reasonable limit here. Let's filter out those records where the *minimum_nights* is greater then 365:

# COMMAND ----------

minNightsDF = posPricesDF.filter(col("minimum_nights") <= 365)

display(minNightsDF)

# COMMAND ----------

# MAGIC %md ### Nulls
# MAGIC 
# MAGIC There are a lot of different ways to handle null values. Sometimes, null can actually be a key indicator of the thing you are trying to predict (e.g. if you don't fill in certain portions of a form, probability of it getting approved decreases).
# MAGIC 
# MAGIC Some ways to handle nulls:
# MAGIC * Drop any records that contain nulls
# MAGIC * Numeric:
# MAGIC   * Replace them with mean/median/zero/etc.
# MAGIC * Categorical:
# MAGIC   * Replace them with the mode
# MAGIC   * Create a special category for null
# MAGIC * Use techniques like ALS which are designed to impute missing values
# MAGIC   
# MAGIC **If you do ANY imputation techniques for categorical/numerical features, you MUST include an additional field specifying that field was imputed.**
# MAGIC 
# MAGIC SparkML's Imputer (covered below) does not support imputation for categorical features.

# COMMAND ----------

# MAGIC %md ### Impute: Cast to Double
# MAGIC 
# MAGIC SparkML's [Imputer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html?highlight=imputer#pyspark.ml.feature.Imputer) requires all fields be of type double. Let's cast all integer fields to double.

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

integerColumns = [x.name for x in minNightsDF.schema.fields if x.dataType == IntegerType()]
doublesDF = minNightsDF

for c in integerColumns:
  doublesDF = doublesDF.withColumn(c, col(c).cast("double"))

columns = "\n - ".join(integerColumns)
print(f"Columns converted from Integer to Double:\n - {columns}")

# COMMAND ----------

# MAGIC %md Add in dummy variable if we will impute any value.

# COMMAND ----------

from pyspark.sql.functions import when

imputeCols = [
  "bedrooms",
  "bathrooms",
  "beds", 
  "review_scores_rating",
  "review_scores_accuracy",
  "review_scores_cleanliness",
  "review_scores_checkin",
  "review_scores_communication",
  "review_scores_location",
  "review_scores_value"
]

for c in imputeCols:
  doublesDF = doublesDF.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))

# COMMAND ----------

display(doublesDF.describe())

# COMMAND ----------

# MAGIC %md ### Transformers and Estimators
# MAGIC 
# MAGIC **Transformer**: Accepts a DataFrame as input, and returns a new DataFrame with one or more columns appended to it. Transformers do not learn any parameters from your
# MAGIC data and simply apply rule-based transformations to either prepare data for model training or generate predictions using a trained MLlib model. They have a `.transform()` method.
# MAGIC 
# MAGIC **Estimator**: Learns (or "fits") parameters from your DataFrame via a `.fit()` method and returns a Model, which is a transformer.

# COMMAND ----------

from pyspark.ml.feature import Imputer

imputer = Imputer(strategy="median", inputCols=imputeCols, outputCols=imputeCols)

imputerModel = imputer.fit(doublesDF)
imputedDF = imputerModel.transform(doublesDF)

# COMMAND ----------

# MAGIC %md
# MAGIC OK, our data is cleansed now. Let's save this DataFrame to Delta so that we can start building models with it.

# COMMAND ----------

outputPath = userhome + "/machine-learning/airbnb-cleansed.delta"

imputedDF.write.format("delta").mode("overwrite").save(outputPath)

# COMMAND ----------

# MAGIC %md ## What's needed for the lab
# MAGIC * [Train-test split](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets)
# MAGIC * [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation)

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://files.training.databricks.com/images/301/TrainTestSplit.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://files.training.databricks.com/images/301/rmse.png)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
