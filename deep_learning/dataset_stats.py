import random
import argparse
import numpy as np
import dnn_functions as dnn
import cv2
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default="stats.csv", help="Output file to write the dataset statistics. Leave at default as other csv files may corrupt the normal functioning of other components")
args = parser.parse_args()

all_data = dnn.read_datasets()

im_mean = [0.0, 0.0, 0.0]
im_std = [0.0, 0.0, 0.0]

r, c, w, h = [], [], [], []

samples = len(all_data)

for data in all_data:
    for i in range(3):
        img = cv2.imread(data[0], -1)
        im_mean[i] += np.mean(img[:,:,i])
        im_std[i] += np.std(img[:,:,i])

    #if data[1] == 1:
    r.append(data[2])
    c.append(data[3])
    w.append(data[4])
    h.append(data[5])
        
im_mean = [i/samples for i in im_mean]
im_std = [i/samples for i in im_std]

r_mean = np.mean(r)
r_std = np.std(r)
c_mean = np.mean(c)
c_std = np.std(c)
w_mean = np.mean(w)
w_std = np.std(w)
h_mean = np.mean(h)
h_std = np.std(h)

print("Results:")
print("Image: Mean: ", im_mean, "\t std: ", im_std) 
print("r_m ", r_mean, "r_std: ", r_std,"c_m ", c_mean, "c_std: ", c_std) 
print("w_m ", w_mean, "w_std: ", w_std,"h_m ", h_mean, "h_std: ", h_std) 

statistics_file = args.output
with open(statistics_file, 'w') as f:
    # Im_mean[0, 1, 2], im_std[0, 1, 2], r_m, r_std, c_m, c_std, w_m, w_std, h_m, h_std
    stats = "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f" %(im_mean[0], im_mean[1], im_mean[2], im_std[0], im_std[1], im_std[2], r_mean, r_std, c_mean, c_std, w_mean, w_std, h_mean, h_std)
    f.write(stats)
