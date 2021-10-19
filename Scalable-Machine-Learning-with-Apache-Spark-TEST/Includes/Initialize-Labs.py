# Databricks notebook source
testResults = dict()

def toHash(value):
  from pyspark.sql.functions import hash
  from pyspark.sql.functions import abs
  values = [(value,)]
  return spark.createDataFrame(values, ["value"]).select(abs(hash("value")).cast("int")).first()[0]

def clearYourResults(passedOnly = True):
  whats = list(testResults.keys())
  for what in whats:
    passed = testResults[what][0]
    if passed or passedOnly == False : del testResults[what]

def validateYourSchema(what, df, expColumnName, expColumnType = None):
  label = "{}:{}".format(expColumnName, expColumnType)
  key = "{} contains {}".format(what, label)

  try:
    actualType = df.schema[expColumnName].dataType.typeName()
    
    if expColumnType == None: 
      testResults[key] = (True, "validated")
      print("""{}: validated""".format(key))
    elif actualType == expColumnType:
      testResults[key] = (True, "validated")
      print("""{}: validated""".format(key))
    else:
      answerStr = "{}:{}".format(expColumnName, actualType)
      testResults[key] = (False, answerStr)
      print("""{}: NOT matching ({})""".format(key, answerStr))
  except:
      testResults[what] = (False, "-not found-")
      print("{}: NOT found".format(key))
      
def validateYourAnswer(what, expectedHash, answer):
  # Convert the value to string, remove new lines and carriage returns and then escape quotes
  if (answer == None): answerStr = "null"
  elif (answer is True): answerStr = "true"
  elif (answer is False): answerStr = "false"
  else: answerStr = str(answer)

  hashValue = toHash(answerStr)
  
  if (hashValue == expectedHash):
    testResults[what] = (True, answerStr)
    print("""{} was correct, your answer: {}""".format(what, answerStr))
  else:
    testResults[what] = (False, answerStr)
    print("""{} was NOT correct, your answer: {}""".format(what, answerStr))

def summarizeYourResults():
  html = """<html><body><div style="font-weight:bold; font-size:larger; border-bottom: 1px solid #f0f0f0">Your Answers</div><table style='margin:0'>"""

  whats = list(testResults.keys())
  whats.sort()
  for what in whats:
    passed = testResults[what][0]
    answer = testResults[what][1]
    color = "green" if (passed) else "red" 
    passFail = "passed" if (passed) else "FAILED" 
    html += """<tr style='font-size:larger; white-space:pre'>
                  <td>{}:&nbsp;&nbsp;</td>
                  <td style="color:{}; text-align:center; font-weight:bold">{}</td>
                  <td style="white-space:pre; font-family: monospace">&nbsp;&nbsp;{}</td>
                </tr>""".format(what, color, passFail, answer)
  html += "</table></body></html>"
  displayHTML(html)

def logYourTest(path, name, value):
  value = float(value)
  if "\"" in path: raise ValueError("The name cannot contain quotes.")
  
  dbutils.fs.mkdirs(path)

  csv = """ "{}","{}" """.format(name, value).strip()
  file = "{}/{}.csv".format(path, name).replace(" ", "-").lower()
  dbutils.fs.put(file, csv, True)

def loadYourTestResults(path):
  from pyspark.sql.functions import col
  return spark.read.schema("name string, value double").csv(path)

def loadYourTestMap(path):
  rows = loadYourTestResults(path).collect()
  
  map = dict()
  for row in rows:
    map[row["name"]] = row["value"]
  
  return map

None

# COMMAND ----------

# %scala
# import org.apache.spark.sql.DataFrame

# val testResults = scala.collection.mutable.Map[String, (Boolean, String)]()

# def toHash(value:String):Int = {
#   import org.apache.spark.sql.functions.hash
#   import org.apache.spark.sql.functions.abs
#   spark.createDataset(List(value)).select(abs(hash($"value")).cast("int")).as[Int].first()
# }

# def clearYourResults(passedOnly:Boolean = true):Unit = {
#   val whats = testResults.keySet.toSeq.sorted
#   for (what <- whats) {
#     val passed = testResults(what)._1
#     if (passed || passedOnly == false) testResults.remove(what)
#   }
# }

# def validateYourSchema(what:String, df:DataFrame, expColumnName:String, expColumnType:String = null):Unit = {
#   val label = s"$expColumnName:$expColumnType"
#   val key = s"$what contains $label"
  
#   try{
#     val actualTypeTemp = df.schema(expColumnName).dataType.typeName
#     val actualType = if (actualTypeTemp.startsWith("decimal")) "decimal" else actualTypeTemp
    
#     if (expColumnType == null) {
#       testResults.put(key,(true, "validated"))
#       println(s"""$key: validated""")
      
#     } else if (actualType == expColumnType) {
#       val answerStr = "%s:%s".format(expColumnName, actualType)
#       testResults.put(key,(true, "validated"))
#       println(s"""$key: validated""")
      
#     } else {
#       val answerStr = "%s:%s".format(expColumnName, actualType)
#       testResults.put(key,(false, answerStr))
#       println(s"""$key: NOT matching ($answerStr)""")
#     }
#   } catch {
#     case e:java.lang.IllegalArgumentException => {
#       testResults.put(key,(false, "-not found-"))
#       println(s"$key: NOT found")
#     }
#   }
# }

# def validateYourAnswer(what:String, expectedHash:Int, answer:Any):Unit = {
#   // Convert the value to string, remove new lines and carriage returns and then escape quotes
#   val answerStr = if (answer == null) "null" 
#   else answer.toString

#   val hashValue = toHash(answerStr)

#   if (hashValue == expectedHash) {
#     testResults.put(what,(true, answerStr))
#     println(s"""$what was correct, your answer: ${answerStr}""")
#   } else{
#     testResults.put(what,(false, answerStr))
#     println(s"""$what was NOT correct, your answer: ${answerStr}""")
#   }
# }

# def summarizeYourResults():Unit = {
#   var html = """<html><body><div style="font-weight:bold; font-size:larger; border-bottom: 1px solid #f0f0f0">Your Answers</div><table style='margin:0'>"""

#   val whats = testResults.keySet.toSeq.sorted
#   for (what <- whats) {
#     val passed = testResults(what)._1
#     val answer = testResults(what)._2
#     val color = if (passed) "green" else "red" 
#     val passFail = if (passed) "passed" else "FAILED" 
#     html += s"""<tr style='font-size:larger; white-space:pre'>
#                   <td>${what}:&nbsp;&nbsp;</td>
#                   <td style="color:${color}; text-align:center; font-weight:bold">${passFail}</td>
#                   <td style="white-space:pre; font-family: monospace">&nbsp;&nbsp;${answer}</td>
#                 </tr>"""
#   }
#   html += "</table></body></html>"
#   displayHTML(html)
# }

# def logYourTest(path:String, name:String, value:Double):Unit = {
#   if (path.contains("\"")) throw new IllegalArgumentException("The name cannot contain quotes.")
  
#   dbutils.fs.mkdirs(path)

#   val csv = """ "%s","%s" """.format(name, value).trim()
#   val file = "%s/%s.csv".format(path, name).replace(" ", "-").toLowerCase
#   dbutils.fs.put(file, csv, true)
# }

# def loadYourTestResults(path:String):org.apache.spark.sql.DataFrame = {
#   return spark.read.schema("name string, value double").csv(path)
# }

# def loadYourTestMap(path:String):scala.collection.mutable.Map[String,Double] = {
#   case class TestResult(name:String, value:Double)
#   val rows = loadYourTestResults(path).collect()
  
#   val map = scala.collection.mutable.Map[String,Double]()
#   for (row <- rows) map.put(row.getString(0), row.getDouble(1))
  
#   return map
# }

# displayHTML("""
#   <div>Initializing lab environment:</div>
#   <li>Declared <b style="color:green">clearYourResults(<i>passedOnly:Boolean=true</i>)</b></li>
#   <li>Declared <b style="color:green">validateYourSchema(<i>what:String, df:DataFrame, expColumnName:String, expColumnType:String</i>)</b></li>
#   <li>Declared <b style="color:green">validateYourAnswer(<i>what:String, expectedHash:Int, answer:Any</i>)</b></li>
#   <li>Declared <b style="color:green">summarizeYourResults()</b></li>
#   <li>Declared <b style="color:green">logYourTest(<i>path:String, name:String, value:Double</i>)</b></li>
#   <li>Declared <b style="color:green">loadYourTestResults(<i>path:String</i>)</b> returns <b style="color:green">DataFrame</b></li>
#   <li>Declared <b style="color:green">loadYourTestMap(<i>path:String</i>)</b> returns <b style="color:green">Map[String,Double]</b></li>
# """)

# COMMAND ----------

displayHTML("All done!")

