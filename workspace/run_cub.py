'''
/* Copyright 2021-2022 Enflame. All Rights Reserved.
 *
 * @file    run_cublas.py
 * @brief   To run benchmarking test for cudnn API
 *
 * @author  ashish(CAI)
 * @date    2021-12-17
 * @version V1.0
 * @par     Copyright (c)
 *          Enflame Tech Company.
 * @par     History:
 */
'''
import glob
import os 
import argparse
import subprocess
import tabulate
from datetime import datetime
import sqlite3

# creating object files 
list_cpp = glob.glob("*.cpp")
print(list_cpp)
for cpp in list_cpp :
  file_name = cpp.split(".")[0]
  os.system("g++ -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include -o {file}.o -c {cpp_file}".format(file = file_name, cpp_file = cpp))

# creating executables
list_obj = glob.glob("*.o")
print(list_obj)
os.system("mkdir Executables")
for obj in list_obj :
  file_name = obj.split(".")[0]
  os.system("/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_70,code=sm_70 -o Executables/{executable} {object} -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include-L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/ppc64le-linux/lib -lcublas -lcudnn -lstdc++ -lm".format(executable = file_name, object = obj))

# Reading config file     
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--configfile", required=True, help="config file name")
args = vars(ap.parse_args())
cfg_path = args['configfile']
config = open(cfg_path, "r")

Table = []
commands = []
total_cases = 0
passed_cases = 0

con=sqlite3.connect('new.db')
print("database connected")

cur=con.cursor()

cur.execute("CREATE TABLE product( cuBLAS_api TEXT, Latency TEXT,throughput TEXT, status1 TEXT, testL TEXT)");


# Running executables            
for cmd in config:
  commands.append(cmd)
  cmd = "./Executables/" + cmd
  os.system("cd Executables")
  total_cases += 1
  
  output = []
  status = ""
  lines = []
  if (os.system(cmd + " >> output.txt") == 0):
    status = "PASSED"
    passed_cases += 1
    proc = subprocess.Popen([cmd], stdout = subprocess.PIPE, shell = True)
    (out, err) = proc.communicate()
    output = str(out).split("\\n")


  else :
    status = "FAILED"

  summary = {"cuBLAS API": "", "Latency": "", "Throughput": "", "Test Level": "", "Status": status}
  cur.execute("INSERT INTO product(status1) VALUES(?)",(status,))

  arguments = cmd.split(" ")
  for line in arguments :
    if ("./" in line) :
      executable = line.split("/")[2]
      summary["cuBLAS API"] = executable.split("_")[1]
      cub=executable.split("_")[1]
      cur.execute("INSERT INTO product( cuBLAS_api) VALUES(?)",(cub,))
    elif ("-L" in line) :
      summary["Test Level"] = line.split("-")[1]
      cub1=line.split("-")[1]
      cur.execute("INSERT INTO product( testL) VALUES(?)",(cub1,))

  for line in output :
    if ("Latency" in line) :
      summary["Latency"] = line.split(": ")[1]
      cub2 = line.split(": ")[1]
      cur.execute("INSERT INTO product(Latency) VALUES(?)",(cub2,))
    elif ("Throughput" in line) :
      summary["Throughput"] = line.split(": ")[1]
      cub3=line.split(": ")[1]
      cur.execute("INSERT INTO product(throughput) VALUES(?)",(cub3,))
  Table.append(summary)

con.commit()

print("cuBLAS_api \t latency \t throughput \t testlevel \t status\n")
cursor = cur.execute("SELECT * FROM product");
for row in cursor : 
  print(row[0],"    \t ",row[1],"    \t ",row[2],"    \t ",row[3],"    \t ",row[4], "\n")
  
  
  
con.close()
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
rows =  [x.values() for x in Table]
print("\n")
print(tabulate.tabulate(rows, header))

failed_cases = total_cases - passed_cases
passed_percentage = (passed_cases * 100) / total_cases

print("\n\n[{passed}/{total} PASSED]".format(passed = passed_cases, total = total_cases))
print("{percent}% tests passed, {failed} tests failed out of {total}".format(percent = passed_percentage, failed = failed_cases, total = total_cases))










