import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--testfile", required=True, help="file name *.test")
ap.add_argument("-n", "--name", required=True)
ap.add_argument("-b", "--batch_size", required=True)
ap.add_argument("-d", "--channel", required=True)
ap.add_argument("-hh", "--height", required=True)
ap.add_argument("-w", "--width", required=True)
args = vars(ap.parse_args())

cfg_path = args['testfile']
name= args['name']
batch_size = int(args['batch_size'])
channels = int(args['channel'])
height = int(args['height'])
width = int(args['width'])

f  = open(cfg_path+".test", "w")
f.write('OP ='+str(name)+"\n")
f.write("BATCH_SIZE = "+str(batch_size)+"\n")
f.write("N_OF_CHANNELS = "+str(channels)+"\n")
f.write("INPUT_HEIGHT = "+str(height)+"\n")
f.write("INPUT_WIDTH = "+str(width))
f.close()

f = open(cfg_path+"_data.txt", "w")
rand_arr = np.random.randint(0,255,size=(batch_size * channels * height * width))
print(rand_arr)

for i in rand_arr:
    f.write(str(i)+" ")
f.close()




