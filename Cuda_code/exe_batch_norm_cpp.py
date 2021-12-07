import argparse
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--testfile", default=" ", help="path to config")
ap.add_argument("-d", "--data", default=" ", help="path to input data")

args = vars(ap.parse_args())

cfg_path = args['testfile']
print("cfg path is ",cfg_path)

if cfg_path == " ":
    print("Config path is missing")

f = open(cfg_path, "r")
cfg_line = f.read()
cfg_line = cfg_line.split('\n')

N, C, H, W = None, None, None, None
i = 1
print("cfg line is ",cfg_line)
for p in cfg_line:
    p = p.split(' ')
    
    if p[-1] != '':
        if i == 1:
            Name = p[-1]
        elif i == 2:
            N = p[-1]
        elif i == 3:
            C = p[-1]
        elif i == 4:
            H = p[-1]
        elif i == 5:
            W = p[-1]
    
    i += 1


if args['data'] == " ":
    subprocess.run(["./batchnormalization", N, C, H, W])
else:
    subprocess.run(["./batchnormalization", N, C, H, W, args['data']])

