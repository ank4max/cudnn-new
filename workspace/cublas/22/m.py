import glob
import os 
import argparse
import subprocess
import tabulate
from datetime import datetime
import sqlite3
import json
import pandas as pd

# creating object files 
list_cpp = glob.glob("*.cpp")
print(list_cpp)
for cpp in list_cpp :
  file_name = cpp.split(".")[0]
  os.system("g++ -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include -o {file}.o -c {cpp_file}".format(file = file_name, cpp_file = cpp))

# creating executables
list_obj = glob.glob("*.o")
print(list_obj)
os.system("sudo mkdir Executables")
for obj in list_obj :
  file_name = obj.split(".")[0]
  os.system("/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_70,code=sm_70 -o Executables/{executable} {object} -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include-L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/ppc64le-linux/lib -lcublas -lcudnn -lstdc++ -lm".format(executable = file_name, object = obj))

# Reading config file     
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--configfile", required=True, help="config file name")
args = vars(ap.parse_args())
cfg_path = args['configfile']
config = open(cfg_path, "r")

summary_table = []
commands = []
total_cases = 0
passed_cases = 0

# setting up conection to connect to database

connection = sqlite3.connect('summary.db')
if (connection)
  print("database connected")
else
  print("Failed to connect to database")
  
#table creation using cursor command
cursor = connection.cursor()
cursor.execute("CREATE TABLE SUMMARY (COMMAND TEXT, LATENCY TEXT, THROUGHPUT TEXT, TEST_LEVEL TEXT, STATUS TEXT)");

json_summary = {}

# Running executables            
for cmd in config:
  commands.append(cmd)
  #cmd = "./Executables/" + cmd
  os.system("cd Executables")
  total_cases += 1
  
  output = []
  status = ""
  lines = []
  
  summary = {"Command": "", "Latency": "", "Throughput": "", "Test_Level": "", "Status": ""}
  summary["Command"] = cmd
  summary["Test_Level"] = "L" + summary["Command"].split("-L")[1].split(" ")[0]
  
  API = summary["Command"].split("_")[1]

  if (os.system(cmd + " >> output.txt") == 0):
    summary["Status"] = "PASSED"
    passed_cases += 1
    proc = subprocess.Popen([cmd], stdout = subprocess.PIPE, shell = True)
    (out, err) = proc.communicate()
    output = str(out).split("\\n")

  else :
    summary["Status"] = "FAILED"
  
  for line in output :
    if ("Latency" in line) :
      summary["Latency"] = line.split(": ")[1]
    elif ("Throughput" in line) :
      summary["Throughput"] = line.split(": ")[1]
  
  if API in dict.keys():
      json_summary[API] = [summary]
  else:
      json_summary[API].append(summary)
  
  Table.append(summary)

  # values insertion in table
  cursor.execute("INSERT INTO SUMMARY (COMMAND, LATENCY, THROUGHPUT, TEST_LEVEL, STATUS) VALUES(?, ?, ?, ?, ?)", \
              (summary["Command"], summary["Latency"], summary["Throughput"], summary["Test_Level"], summary["Status"]))

# saving changes in table  
connection.commit()
connection.close()

with open("Summary.json",'w') as json_file :
    json.dump(json_summary,json_file)

with open("Summary.json",'r') as json_file :
    json_Summary = json.load(json_file)
    print(json_Summary)
    print("printed json data")

# printng database table
connection = sqlite3.connect('summary.db')
cursor = connection.cursor()

print("COMMAND \t\t\t LATENCY \t THROUGHPUT \t TEST_LEVEL \t STATUS\n")
rows = cursor.execute("SELECT * FROM SUMMARY");
for values in rows : 
  print(values[0],"    ", values[1],"\t", values[2]," \t", values[3],"    ", values[4], "\n")

# open json data
with open("Summary.json") as f :
  data = json.load(f)

#create a DataFrame From the Json data
df = pd.DataFrame(data)

df.to_sql("SUMMARY",connection)

connection.close()

print("\n\nExecuted below cuBLAS Test Cases")
print("===============================")
for cmd in commands :
  print(cmd.split("\n")[0])

# Printing table
print("\n\n\n=============================")
print("Test Result Summary of cuBLAS")
print("=============================")
print("Executed on: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

header = Table[0].keys()
rows = [x.values() for x in Table]
print("\n")
print(tabulate.tabulate(rows, header))

failed_cases = total_cases - passed_cases
passed_percentage = (passed_cases * 100) / total_cases

print("\n\n[{passed}/{total} PASSED]".format(passed = passed_cases, total = total_cases))
print("{percent}% tests passed, {failed} tests failed out of {total}".format(percent = passed_percentage, failed = failed_cases, total = total_cases))
