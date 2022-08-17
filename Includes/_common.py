# Databricks notebook source
# MAGIC %pip install \
# MAGIC git+https://github.com/databricks-academy/dbacademy-gems@c3032c2df47472f1600d368523f052d2920b406d \
# MAGIC git+https://github.com/databricks-academy/dbacademy-rest@e729b6dbb566de2958cba60fe4bd50e1b9e7f25b \
# MAGIC git+https://github.com/databricks-academy/dbacademy-helper@fd1619a8b6f22adb3b7b54e2897cbdc5c3f161a4 \
# MAGIC --quiet --disable-pip-version-check

# COMMAND ----------

# MAGIC %run ./_dataset_index

# COMMAND ----------

from dbacademy_gems import dbgems
from dbacademy_helper import DBAcademyHelper, Paths

helper_arguments = {
    "course_code" : "smlwas",          # The abreviated version of the course
    "course_name" : "scalable-machine-learning-with-apache-spark",      # The full name of the course, hyphenated
    "data_source_name" : "scalable-machine-learning-with-apache-spark", # Should be the same as the course
    "data_source_version" : "v02",     # New courses would start with 01
    "enable_streaming_support": False, # This couse uses stream and thus needs checkpoint directories
    "install_min_time" : "2 min",      # The minimum amount of time to install the datasets (e.g. from Oregon)
    "install_max_time" : "5 min",      # The maximum amount of time to install the datasets (e.g. from India)
    "remote_files": remote_files,      # The enumerated list of files in the datasets
}

