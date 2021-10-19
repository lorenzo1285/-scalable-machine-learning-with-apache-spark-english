# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Distributed K-Means
# MAGIC 
# MAGIC In this notebook, we are going to use K-Means to cluster our data. We will be using the Iris dataset, which has labels (the type of iris), but we will only use the labels to evaluate the model, not to train it. 
# MAGIC 
# MAGIC At the end, we will look at how it is implemented in the distributed setting.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Build a K-Means model
# MAGIC  - Analyze the computation and communication of K-Means in a distributed setting

# COMMAND ----------

from sklearn.datasets import load_iris
import pandas as pd

# Load in a Dataset from sklearn and convert to a Spark DataFrame
iris = load_iris()
iris_pd = pd.concat([pd.DataFrame(iris.data, columns=iris.feature_names), pd.DataFrame(iris.target, columns=["label"])], axis=1)
irisDF = spark.createDataFrame(iris_pd)
display(irisDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that we have four values as "features".  We'll reduce those down to two values (for visualization purposes) and convert them to a `DenseVector`.  To do that we'll use the `VectorAssembler`. 

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=["sepal length (cm)", "sepal width (cm)"], outputCol="features")
irisTwoFeaturesDF = vecAssembler.transform(irisDF)
display(irisTwoFeaturesDF)

# COMMAND ----------

from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=3, seed=221, maxIter=20)

#  Call fit on the estimator and pass in irisTwoFeaturesDF
model = kmeans.fit(irisTwoFeaturesDF)

# Obtain the clusterCenters from the KMeansModel
centers = model.clusterCenters()

# Use the model to transform the DataFrame by adding cluster predictions
transformedDF = model.transform(irisTwoFeaturesDF)

print(centers)

# COMMAND ----------

modelCenters = []
iterations = [0, 2, 4, 7, 10, 20]
for i in iterations:
    kmeans = KMeans(k=3, seed=221, maxIter=i)
    model = kmeans.fit(irisTwoFeaturesDF)
    modelCenters.append(model.clusterCenters())   

# COMMAND ----------

print("modelCenters:")
for centroids in modelCenters:
  print(centroids)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's visualize how our clustering performed against the true labels of our data.
# MAGIC 
# MAGIC Remember: K-means doesn't use the true labels when training, but we can use them to evaluate. 
# MAGIC 
# MAGIC Here, the star marks the cluster center.

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def prepareSubplot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor="#999999", 
                gridWidth=1.0, subplots=(1, 1)):
    """Template for generating the plot layout."""
    fig, axList = plt.subplots(subplots[0], subplots[1], figsize=figsize, facecolor="white", 
                               edgecolor="white")
    if not isinstance(axList, np.ndarray):
        axList = np.array([axList])
    
    for ax in axList.flatten():
        ax.axes.tick_params(labelcolor="#999999", labelsize="10")
        for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
            axis.set_ticks_position("none")
            axis.set_ticks(ticks)
            axis.label.set_color("#999999")
            if hideLabels: axis.set_ticklabels([])
        ax.grid(color=gridColor, linewidth=gridWidth, linestyle="-")
        map(lambda position: ax.spines[position].set_visible(False), ["bottom", "top", "left", "right"])
        
    if axList.size == 1:
        axList = axList[0]  # Just return a single axes object for a regular plot
    return fig, axList

# COMMAND ----------

data = irisTwoFeaturesDF.select("features", "label").collect()
features, labels = zip(*data)

x, y = zip(*features)
centers = modelCenters[5]
centroidX, centroidY = zip(*centers)
colorMap = "Set1"

fig, ax = prepareSubplot(np.arange(-1, 1.1, .4), np.arange(-1, 1.1, .4), figsize=(8,6))
plt.scatter(x, y, s=14**2, c=labels, edgecolors="#8cbfd0", alpha=0.80, cmap=colorMap)
plt.scatter(centroidX, centroidY, s=22**2, marker="*", c="yellow")
cmap = cm.get_cmap(colorMap)

colorIndex = [.5, .99, .0]
for i, (x,y) in enumerate(centers):
    print(cmap(colorIndex[i]))
    for size in [.10, .20, .30, .40, .50]:
        circle1=plt.Circle((x,y),size,color=cmap(colorIndex[i]), alpha=.10, linewidth=2)
        ax.add_artist(circle1)

ax.set_xlabel("Sepal Length"), ax.set_ylabel("Sepal Width")
fig

# COMMAND ----------

# MAGIC %md
# MAGIC In addition to seeing the overlay of the clusters at each iteration, we can see how the cluster centers moved with each iteration (and what our results would have looked like if we used fewer iterations).

# COMMAND ----------

x, y = zip(*features)

oldCentroidX, oldCentroidY = None, None

fig, axList = prepareSubplot(np.arange(-1, 1.1, .4), np.arange(-1, 1.1, .4), figsize=(11, 15),
                             subplots=(3, 2))
axList = axList.flatten()

for i,ax in enumerate(axList[:]):
    ax.set_title("K-means for {0} iterations".format(iterations[i]), color="#999999")
    centroids = modelCenters[i]
    centroidX, centroidY = zip(*centroids)
    
    ax.scatter(x, y, s=10**2, c=labels, edgecolors="#8cbfd0", alpha=0.80, cmap=colorMap, zorder=0)
    ax.scatter(centroidX, centroidY, s=16**2, marker="*", c="yellow", zorder=2)
    if oldCentroidX and oldCentroidY:
      ax.scatter(oldCentroidX, oldCentroidY, s=16**2, marker="*", c="grey", zorder=1)
    cmap = cm.get_cmap(colorMap)
    
    colorIndex = [.5, .99, 0.]
    for i, (x1,y1) in enumerate(centroids):
      print(cmap(colorIndex[i]))
      circle1=plt.Circle((x1,y1),.35,color=cmap(colorIndex[i]), alpha=.40)
      ax.add_artist(circle1)
    
    ax.set_xlabel("Sepal Length"), ax.set_ylabel("Sepal Width")
    oldCentroidX, oldCentroidY = centroidX, centroidY

plt.tight_layout()

fig

# COMMAND ----------

# MAGIC %md So let's take a look at what's happening here in the distributed setting.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://files.training.databricks.com/images/Mapstage.png" height=200px>

# COMMAND ----------

# MAGIC %md <img src="https://files.training.databricks.com/images/Mapstage2.png" height=500px>

# COMMAND ----------

# MAGIC %md <img src="https://files.training.databricks.com/images/ReduceStage.png" height=500px>

# COMMAND ----------

# MAGIC %md <img src="https://files.training.databricks.com/images/Communication.png" height=500px>

# COMMAND ----------

# MAGIC %md ## Take Aways
# MAGIC 
# MAGIC When designing/choosing distributed ML algorithms
# MAGIC * Communication is key!
# MAGIC * Consider your data/model dimensions & how much data you need.
# MAGIC * Data partitioning/organization is important.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
