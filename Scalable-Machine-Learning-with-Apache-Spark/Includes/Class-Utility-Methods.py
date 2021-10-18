# Databricks notebook source

# # Databricks notebook source
# def get_cloud():
#   with open("/databricks/common/conf/deploy.conf") as f:
#     for line in f:
#       if "databricks.instance.metadata.cloudProvider" in line and "\"GCP\"" in line: return "GCP"
#       elif "databricks.instance.metadata.cloudProvider" in line and "\"AWS\"" in line: return "AWS"
#       elif "databricks.instance.metadata.cloudProvider" in line and "\"Azure\"" in line: return "MSA"

#############################################
# TAG API FUNCTIONS
#############################################

# Get all tags
def getTags() -> dict: 
  return sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(
    dbutils.entry_point.getDbutils().notebook().getContext().tags()
  )

# Get a single tag's value
def getTag(tagName: str, defaultValue: str = None) -> str:
  values = getTags()[tagName]
  try:
    if len(values) > 0:
      return values
  except:
    return defaultValue

#############################################
# Get Databricks runtime major and minor versions
#############################################

def getDbrMajorAndMinorVersions() -> (int, int):
  import os
  dbrVersion = os.environ["DATABRICKS_RUNTIME_VERSION"]
  dbrVersion = dbrVersion.split(".")
  return (int(dbrVersion[0]), int(dbrVersion[1]))

# Get Python version
def getPythonVersion() -> str:
  import sys
  pythonVersion = sys.version[0:sys.version.index(" ")]
  spark.conf.set("com.databricks.training.python-version", pythonVersion)
  return pythonVersion

#############################################
# USER, USERNAME, AND USERHOME FUNCTIONS
#############################################

def get_cloud():
  with open("/databricks/common/conf/deploy.conf") as f:
    for line in f:
      if "databricks.instance.metadata.cloudProvider" in line and "\"GCP\"" in line: return "GCP"
      elif "databricks.instance.metadata.cloudProvider" in line and "\"AWS\"" in line: return "AWS"
      elif "databricks.instance.metadata.cloudProvider" in line and "\"Azure\"" in line: return "MSA"
              
# Get the user's username
def getUsername() -> str:
  return spark.sql("SELECT current_user()").first()[0]

# Get the user's userhome
def getUserhome() -> str:
  username = getUsername()
  return f"file:///dbfs/user/{username}/dbacademy"

  # cloud = get_cloud()
  # if cloud == "GCP":
  #   return f"file:///dbacademy/{username}"
  # else:
  #  return f"file:///dbfs/user/{username}/dbacademy"

def getModuleName() -> str: 
  # This will/should fail if module-name is not defined in the Classroom-Setup notebook
  return spark.conf.get("com.databricks.training.module-name")

def getLessonName() -> str:
  # If not specified, use the notebook's name.
  return dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None).split("/")[-1]

def getCourseDir() -> str:
  import re
  moduleName = re.sub(r"[^a-zA-Z0-9]", "_", getModuleName()).lower()
  courseDir = f"{getUserhome()}/{moduleName}"
  return courseDir.replace("__", "_").replace("__", "_").replace("__", "_").replace("__", "_")
  
def getWorkingDir() -> str:
  import re
  course_dir = getCourseDir()
  lessonName = re.sub(r"[^a-zA-Z0-9]", "_", getLessonName()).lower()
  workingDir = f"{getCourseDir()}/{lessonName}"
  return workingDir.replace("__", "_").replace("__", "_").replace("__", "_").replace("__", "_")


#############################################
# VERSION ASSERTION FUNCTIONS
#############################################

# When migrating DBR versions this should be one
# of the only two places that needs to be updated
latestDbrMajor = 7
latestDbrMinor = 0

  # Assert an appropriate Databricks Runtime version
def assertDbrVersion(expected:str, latestMajor:int=latestDbrMajor, latestMinor:int=latestDbrMinor, display:bool = True):
  
  expMajor = latestMajor
  expMinor = latestMinor
  
  if expected and expected != "{{dbr}}":
    expMajor = int(expected.split(".")[0])
    expMinor = int(expected.split(".")[1])

  (major, minor) = getDbrMajorAndMinorVersions()

  if (major < expMajor) or (major == expMajor and minor < expMinor):
    msg = f"This notebook must be run on DBR {expMajor}.{expMinor} or newer. Your cluster is using {major}.{minor}. You must update your cluster configuration before proceeding."

    raise AssertionError(msg)
    
  if major != expMajor or minor != expMinor:
    html = f"""
      <div style="color:red; font-weight:bold">WARNING: This notebook was tested on DBR {expMajor}.{expMinor}, but we found DBR {major}.{minor}.</div>
      <div style="font-weight:bold">Using an untested DBR may yield unexpected results and/or various errors</div>
      <div style="font-weight:bold">Please update your cluster configuration and/or <a href="https://academy.databricks.com/" target="_blank">download a newer version of this course</a> before proceeding.</div>
    """

  else:
    html = f"Running on <b>DBR {major}.{minor}</b>"
  
  if display:
    displayHTML(html)
  else:
    print(html)
  
  return f"{major}.{minor}"

# Assert that the Databricks Runtime is ML version
# def assertIsMlRuntime(testValue: str = None):

#   if testValue is not None: sourceValue = testValue
#   else: sourceValue = getRuntimeVersion()

#   if "-ml-" not in sourceValue:
#     raise AssertionError(f"This notebook must be ran on a Databricks ML Runtime, found {sourceValue}.")

    
############################################
# USER DATABASE FUNCTIONS
############################################

def getDatabaseName(courseType:str, username:str, moduleName:str, lessonName:str) -> str:
  import re
  langType = "p" # for python
  databaseName = username + "_" + moduleName + "_" + lessonName + "_" + langType + courseType
  databaseName = databaseName.lower()
  databaseName = re.sub("[^a-zA-Z0-9]", "_", databaseName)
  return databaseName.replace("__", "_").replace("__", "_").replace("__", "_").replace("__", "_")


# Create a user-specific database
def createUserDatabase(courseType:str, username:str, moduleName:str, lessonName:str) -> str:
  databaseName = getDatabaseName(courseType, username, moduleName, lessonName)

  spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(databaseName))
  spark.sql("USE {}".format(databaseName))

  return databaseName

    
#############################################
# LEGACY TESTING FUNCTIONS
#############################################

# Test results dict to store results
testResults = dict()

# Hash a string value
def toHash(value):
  from pyspark.sql.functions import hash
  from pyspark.sql.functions import abs
  values = [(value,)]
  return spark.createDataFrame(values, ["value"]).select(abs(hash("value")).cast("int")).first()[0]

# Clear the testResults map
def clearYourResults(passedOnly = True):
  whats = list(testResults.keys())
  for what in whats:
    passed = testResults[what][0]
    if passed or passedOnly == False : del testResults[what]

# Validate DataFrame schema
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

# Validate an answer
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

# Summarize results in the testResults dict
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

# Log test results to a file
def logYourTest(path, name, value):
  value = float(value)
  if "\"" in path: raise AssertionError("The name cannot contain quotes.")
  
  dbutils.fs.mkdirs(path)

  csv = """ "{}","{}" """.format(name, value).strip()
  file = "{}/{}.csv".format(path, name).replace(" ", "-").lower()
  dbutils.fs.put(file, csv, True)

# Load test results from log file
def loadYourTestResults(path):
  from pyspark.sql.functions import col
  return spark.read.schema("name string, value double").csv(path)

# Load test results from log file into a dict
def loadYourTestMap(path):
  rows = loadYourTestResults(path).collect()
  
  map = dict()
  for row in rows:
    map[row["name"]] = row["value"]
  
  return map

# ****************************************************************************
# Utility method to determine whether a path exists
# ****************************************************************************

def pathExists(path):
  try:
    dbutils.fs.ls(path)
    return True
  except:
    return False
  
# ****************************************************************************
# Utility method for recursive deletes
# Note: dbutils.fs.rm() does not appear to be truely recursive
# ****************************************************************************

def deletePath(path):
  files = dbutils.fs.ls(path)

  for file in files:
    deleted = dbutils.fs.rm(file.path, True)
    
    if deleted == False:
      if file.is_dir:
        deletePath(file.path)
      else:
        raise IOError("Unable to delete file: " + file.path)
  
  if dbutils.fs.rm(path, True) == False:
    raise IOError("Unable to delete directory: " + path)

# ****************************************************************************
# Utility method to clean up the workspace at the end of a lesson
# ****************************************************************************

def classroomCleanup(daLogger:object, courseType:str, username:str, moduleName:str, lessonName:str, dropDatabase:str): 
  import time

  actions = ""
  
  # Stop any active streams
  for stream in spark.streams.active:
    stream.stop()
    
    # Wait for the stream to stop
    queries = list(filter(lambda query: query.name == stream.name, spark.streams.active))
    
    while (len(queries) > 0):
      time.sleep(5) # Give it a couple of seconds
      queries = list(filter(lambda query: query.name == stream.name, spark.streams.active))

    actions += f"""<li>Terminated stream: <b>{stream.name}</b></li>"""
  
  # Drop all tables from the specified database
  database = getDatabaseName(courseType, username, moduleName, lessonName)
  try:
    tables = spark.sql("show tables from {}".format(database)).select("tableName").collect()
    for row in tables:
      tableName = row["tableName"]
      spark.sql("drop table if exists {}.{}".format(database, tableName))

      # In some rare cases the files don't actually get removed.
      time.sleep(1) # Give it just a second...
      hivePath = "dbfs:/user/hive/warehouse/{}.db/{}".format(database, tableName)
      dbutils.fs.rm(hivePath, True) # Ignoring the delete's success or failure
      
      actions += f"""<li>Dropped table: <b>{tableName}</b></li>"""

  except:
    pass # ignored

  # The database should only be dropped in a "cleanup" notebook, not "setup"
  if dropDatabase: 
    spark.sql("DROP DATABASE IF EXISTS {} CASCADE".format(database))
    
    # In some rare cases the files don't actually get removed.
    time.sleep(1) # Give it just a second...
    hivePath = "dbfs:/user/hive/warehouse/{}.db".format(database)
    dbutils.fs.rm(hivePath, True) # Ignoring the delete's success or failure
    
    actions += f"""<li>Dropped database: <b>{database}</b></li>"""

  # Remove any files that may have been created from previous runs
  path = getWorkingDir(courseType)
  if pathExists(path):
    deletePath(path)

    actions += f"""<li>Removed working directory: <b>{path}</b></li>"""
    
  htmlMsg = "Cleaning up the learning environment..."
  if len(actions) == 0: htmlMsg += "no actions taken."
  else:  htmlMsg += f"<ul>{actions}</ul>"
  displayHTML(htmlMsg)
  
  if dropDatabase: daLogger.logEvent("Classroom-Cleanup-Final")
  else: daLogger.logEvent("Classroom-Cleanup-Preliminary")

  
# Utility method to delete a database  
def deleteTables(database):
  spark.sql("DROP DATABASE IF EXISTS {} CASCADE".format(database))
  
    
# ****************************************************************************
# DatabricksAcademyLogger and Student Feedback
# ****************************************************************************

class DatabricksAcademyLogger:
  
  def logEvent(self, eventId: str, message: str = None):
    import time
    import json
    import requests

    hostname = "https://rqbr3jqop0.execute-api.us-west-2.amazonaws.com/prod"
    
    try:
      username = getUsername().encode("utf-8")
      moduleName = getModuleName().encode("utf-8")
      lessonName = getLessonName().encode("utf-8")
      event = eventId.encode("utf-8")
    
      content = {
        "tags":       dict(map(lambda x: (x[0], str(x[1])), getTags().items())),
        "moduleName": getModuleName(),
        "lessonName": getLessonName(),
        "orgId":      getTag("orgId", "unknown"),
        "username":   getUsername(),
        "eventId":    eventId,
        "eventTime":  f"{int(round(time.time() * 1000))}",
        "language":   getTag("notebookLanguage", "unknown"),
        "notebookId": getTag("notebookId", "unknown"),
        "sessionId":  getTag("sessionId", "unknown"),
        "message":    message
      }
      
      response = requests.post( 
          url=f"{hostname}/logger", 
          json=content,
          headers={
            "Accept": "application/json; charset=utf-8",
            "Content-Type": "application/json; charset=utf-8"
          })
      
    except Exception as e:
      pass

    
def showStudentSurvey():
  html = renderStudentSurvey()
  displayHTML(html);

def renderStudentSurvey():
  username = getUsername().encode("utf-8")
  userhome = getUserhome().encode("utf-8")

  moduleName = getModuleName().encode("utf-8")
  lessonName = getLessonName().encode("utf-8")
  lessonNameUnencoded = getLessonName()
  
  apiEndpoint = "https://rqbr3jqop0.execute-api.us-west-2.amazonaws.com/prod"

  feedbackUrl = f"{apiEndpoint}/feedback";
  
  html = """
  <html>
  <head>
    <script src="https://files.training.databricks.com/static/js/classroom-support.min.js"></script>
    <script>
<!--    
      window.setTimeout( // Defer until bootstrap has enough time to async load
        () => { 
          $("#divComment").css("display", "visible");

          // Emulate radio-button like feature for multiple_choice
          $(".multiple_choicex").on("click", (evt) => {
                const container = $(evt.target).parent();
                $(".multiple_choicex").removeClass("checked"); 
                $(".multiple_choicex").removeClass("checkedRed"); 
                $(".multiple_choicex").removeClass("checkedGreen"); 
                container.addClass("checked"); 
                if (container.hasClass("thumbsDown")) { 
                    container.addClass("checkedRed"); 
                } else { 
                    container.addClass("checkedGreen"); 
                };
                
                // Send the like/dislike before the comment is shown so we at least capture that.
                // In analysis, always take the latest feedback for a module (if they give a comment, it will resend the like/dislike)
                var json = {
                  moduleName: "GET_MODULE_NAME", 
                  lessonName: "GET_LESSON_NAME", 
                  orgId:       "GET_ORG_ID",
                  username:    "GET_USERNAME",
                  language:    "python",
                  notebookId:  "GET_NOTEBOOK_ID",
                  sessionId:   "GET_SESSION_ID",
                  survey: $(".multiple_choicex.checked").attr("value"), 
                  comment: $("#taComment").val() 
                };
                
                $("#vote-response").html("Recording your vote...");

                $.ajax({
                  type: "PUT", 
                  url: "FEEDBACK_URL", 
                  data: JSON.stringify(json),
                  dataType: "json",
                  processData: false
                }).done(function() {
                  $("#vote-response").html("Thank you for your vote!<br/>Please feel free to share more if you would like to...");
                  $("#divComment").show("fast");
                }).fail(function() {
                  $("#vote-response").html("There was an error submitting your vote");
                }); // End of .ajax chain
          });


           // Set click handler to do a PUT
          $("#btnSubmit").on("click", (evt) => {
              // Use .attr("value") instead of .val() - this is not a proper input box
              var json = {
                moduleName: "GET_MODULE_NAME", 
                lessonName: "GET_LESSON_NAME", 
                orgId:       "GET_ORG_ID",
                username:    "GET_USERNAME",
                language:    "python",
                notebookId:  "GET_NOTEBOOK_ID",
                sessionId:   "GET_SESSION_ID",
                survey: $(".multiple_choicex.checked").attr("value"), 
                comment: $("#taComment").val() 
              };

              $("#feedback-response").html("Sending feedback...");

              $.ajax({
                type: "PUT", 
                url: "FEEDBACK_URL", 
                data: JSON.stringify(json),
                dataType: "json",
                processData: false
              }).done(function() {
                  $("#feedback-response").html("Thank you for your feedback!");
              }).fail(function() {
                  $("#feedback-response").html("There was an error submitting your feedback");
              }); // End of .ajax chain
          });
        }, 2000
      );
-->
    </script>    
    <style>
.multiple_choicex > img {    
    border: 5px solid white;
    border-radius: 5px;
}
.multiple_choicex.choice1 > img:hover {    
    border-color: green;
    background-color: green;
}
.multiple_choicex.choice2 > img:hover {    
    border-color: red;
    background-color: red;
}
.multiple_choicex {
    border: 0.5em solid white;
    background-color: white;
    border-radius: 5px;
}
.multiple_choicex.checkedGreen {
    border-color: green;
    background-color: green;
}
.multiple_choicex.checkedRed {
    border-color: red;
    background-color: red;
}
    </style>
  </head>
  <body>
    <h2 style="font-size:28px; line-height:34.3px"><img style="vertical-align:middle" src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"/>What did you think?</h2>
    <p>Please let us know if you liked this notebook, <b>LESSON_NAME_UNENCODED</b></p>
    <div id="feedback" style="clear:both;display:table;">
      <span class="multiple_choicex choice1 thumbsUp" value="positive"><img style="width:100px" src="https://files.training.databricks.com/images/feedback/thumbs-up.png"/></span>
      <span class="multiple_choicex choice2 thumbsDown" value="negative"><img style="width:100px" src="https://files.training.databricks.com/images/feedback/thumbs-down.png"/></span>
      <div id="vote-response" style="color:green; margin:1em 0; font-weight:bold">&nbsp;</div>
      <table id="divComment" style="display:none; border-collapse:collapse;">
        <tr>
          <td style="padding:0"><textarea id="taComment" placeholder="How can we make this lesson better? (optional)" style="height:4em;width:30em;display:block"></textarea></td>
          <td style="padding:0"><button id="btnSubmit" style="margin-left:1em">Send</button></td>
        </tr>
      </table>
    </div>
    <div id="feedback-response" style="color:green; margin-top:1em; font-weight:bold">&nbsp;</div>
  </body>
  </html>
  """

  return (html.replace("GET_MODULE_NAME", getModuleName())
              .replace("GET_LESSON_NAME", getLessonName())
              .replace("GET_ORG_ID", getTag("orgId", "unknown"))
              .replace("GET_USERNAME", getUsername())
              .replace("GET_NOTEBOOK_ID", getTag("notebookId", "unknown"))
              .replace("GET_SESSION_ID", getTag("sessionId", "unknown"))
              .replace("LESSON_NAME_UNENCODED", lessonNameUnencoded)
              .replace("FEEDBACK_URL", feedbackUrl)
         )

# ****************************************************************************
# Facility for advertising functions, variables and databases to the student
# ****************************************************************************
def allDone(advertisements):
  
  functions = dict()
  variables = dict()
  databases = dict()
  
  for key in advertisements:
    if advertisements[key][0] == "f" and spark.conf.get(f"com.databricks.training.suppress.{key}", None) != "true":
      functions[key] = advertisements[key]
  
  for key in advertisements:
    if advertisements[key][0] == "v" and spark.conf.get(f"com.databricks.training.suppress.{key}", None) != "true":
      variables[key] = advertisements[key]
  
  for key in advertisements:
    if advertisements[key][0] == "d" and spark.conf.get(f"com.databricks.training.suppress.{key}", None) != "true":
      databases[key] = advertisements[key]
  
  html = ""
  if len(functions) > 0:
    html += "The following functions were defined for you:<ul style='margin-top:0'>"
    for key in functions:
      value = functions[key]
      html += f"""<li style="cursor:help" onclick="document.getElementById('{key}').style.display='block'">
        <span style="color: green; font-weight:bold">{key}</span>
        <span style="font-weight:bold">(</span>
        <span style="color: green; font-weight:bold; font-style:italic">{value[1]}</span>
        <span style="font-weight:bold">)</span>
        <div id="{key}" style="display:none; margin:0.5em 0; border-left: 3px solid grey; padding-left: 0.5em">{value[2]}</div>
        </li>"""
    html += "</ul>"

  if len(variables) > 0:
    html += "The following variables were defined for you:<ul style='margin-top:0'>"
    for key in variables:
      value = variables[key]
      html += f"""<li style="cursor:help" onclick="document.getElementById('{key}').style.display='block'">
        <span style="color: green; font-weight:bold">{key}</span>: <span style="font-style:italic; font-weight:bold">{value[1]} </span>
        <div id="{key}" style="display:none; margin:0.5em 0; border-left: 3px solid grey; padding-left: 0.5em">{value[2]}</div>
        </li>"""
    html += "</ul>"

  if len(databases) > 0:
    html += "The following database were created for you:<ul style='margin-top:0'>"
    for key in databases:
      value = databases[key]
      html += f"""<li style="cursor:help" onclick="document.getElementById('{key}').style.display='block'">
        Now using the database identified by <span style="color: green; font-weight:bold">{key}</span>: 
        <div style="font-style:italic; font-weight:bold">{value[1]}</div>
        <div id="{key}" style="display:none; margin:0.5em 0; border-left: 3px solid grey; padding-left: 0.5em">{value[2]}</div>
        </li>"""
    html += "</ul>"

  html += "All done!"
  displayHTML(html)

# ****************************************************************************
# Placeholder variables for coding challenge type specification
# ****************************************************************************
class FILL_IN:
  from pyspark.sql.types import Row, StructType
  VALUE = None
  LIST = []
  SCHEMA = StructType([])
  ROW = Row()
  INT = 0
  DATAFRAME = sqlContext.createDataFrame(sc.emptyRDD(), StructType([]))

# ****************************************************************************
# Initialize the logger so that it can be used down-stream
# ****************************************************************************

daLogger = DatabricksAcademyLogger()
daLogger.logEvent("Initialized", "Initialized the Python DatabricksAcademyLogger")

displayHTML("Defining courseware-specific utility methods...")

