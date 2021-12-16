import argparse
import subprocess
from nvitop.core import host, Device, HostProcess, GpuProcess
import os
import psutil
import time

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--testfile", default=" ", help="path to config")
ap.add_argument("-d", "--data", default=" ", help="path to input data")
ap.add_argument("-i", "--iter", default="1000", help="throughput over iterartion")

args = vars(ap.parse_args())

cfg_path = args['testfile']

if cfg_path == " ":
    print("Config path is missing")

f = open(cfg_path, "r")
cfg_line = f.read()
cfg_line = cfg_line.split('\n')

N, C, H, W = None, None, None, None
i = 1
for p in cfg_line:
    p = p.split(' ')
    
    if p[-1] != '':
        if i == 1:
            N = p[-1]
        elif i == 2:
            C = p[-1]
        elif i == 3:
            H = p[-1]
        elif i == 4:
            W = p[-1]
    
    i += 1
    
N=str(N)
C=str(C)
H=str(H)
W=str(W)
device = Device(0)
this_process = GpuProcess(os.getpid(), device)


def call():
    if args['data'] == " ":
        t = 0
        for i in range(int(args['iter'])):
            t1 = time.time()
            process = subprocess.Popen("./batch14 -n2 -c2 -h2 -w1 -atanh".split(" "))
            # print(str(process.pid))
            this_process = GpuProcess(process.pid, device)
            # subprocess.run(['ps'])
            
            t2 = time.time()
            t += (t2-t1)
            this_process.update_gpu_status()
        data = {'Total_number_of_CPUs': psutil.cpu_count(),
            'device/memory_used': float(device.memory_used()) / (1 << 20),  # convert bytes to MiBs
            'cpu_memory_used_in_percent': float(psutil.cpu_percent()),
            'Percentage_available_RAM': psutil.virtual_memory().available * 100 / psutil.virtual_memory().total,
            'virtual_memory_used_in_percent': float(psutil.virtual_memory()[2]),
            'device/gpu_utilization': device.gpu_utilization(),
            'process/used_gpu_memory': float(this_process.gpu_memory()) / (1 << 20),
            'average_time': t/1000
            }  # convert bytes to MiBs
       
    else:
        t = 0
        for i in range(int(args['iter'])):
            t1 = time.time()
            process = subprocess.Popen("./batch14 -n2 -c2 -h2 -w1 -atanh".split(" "))
            # print(str(process.pid))
            this_process = GpuProcess(process.pid, device)
            
            t2 = time.time()
            t += (t2-t1)
            this_process.update_gpu_status()
        data = {'Total_number_of_CPUs': psutil.cpu_count(),
            'device/memory_used': float(device.memory_used()) / (1 << 20),  # convert bytes to MiBs
            'cpu_memory_used_in_percent': float(psutil.cpu_percent()),
            'Percentage_available_RAM': psutil.virtual_memory().available * 100 / psutil.virtual_memory().total,
            'virtual_memory_used_in_percent': float(psutil.virtual_memory()[2]),
            'device/gpu_utilization': device.gpu_utilization(),
            'process/used_gpu_memory': float(this_process.gpu_memory()) / (1 << 20),
            'average_time': t/1000
            }  # convert bytes to MiBs

    return data

data = call()
# print(data)
print("\n\n")
print("-----------------------------===REPORT===-------------------------------------------")
print("Total number of CPUs: ", "\t\t\t", data['Total_number_of_CPUs'])
print("Total CPUs utilized percentage: ", "\t", data['cpu_memory_used_in_percent'])
print("Total CPUs utilized memory (mb): ", "\t", data['device/memory_used'])
print("Percentage of available RAM: ", "\t\t", data['Percentage_available_RAM'])
print("Percentage of used RAM : ", "\t\t", data['virtual_memory_used_in_percent'])
# print("Gpu_utilization : ", "\t\t\t", data['device/gpu_utilization'])
print("process/used_gpu_memory : ", "\t\t", data['process/used_gpu_memory'])
print("Throughput (over 1000 iteration) : ", "\t", data['average_time'])
print("------------------------------------------------------------------------")




