''
/* Copyright 2021-2022 Enflame. All Rights Reserved.
 *
 * @file    run_cudnn.py
 * @brief   To run benchmarking test for cudnn API
 *
 * @author  ashish(CAI)
 * @date    2021-12-13
 * @version V1.0
 * @par     Copyright (c)
 *          Enflame Tech Company.
 * @par     History:
 */
'''

import glob
import os 
import argparse

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

# Running executables            
for cmd in config:
  print(cmd)  
  os.system("cd Executables")
  os.system("./Executables/" + cmd) 
