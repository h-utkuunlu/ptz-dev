import random
import argparse
import numpy as np
import dnn_functions as dnn
import cv2
import csv

parser = argparse.ArgumentParser()
parser.add_argument('dir', help="Directory of positive and negative sample directory to compute the dataset stats")
parser.add_argument('-o', '--output', default="stats", help="Output file to write the dataset statistics. Leave at default as other csv files may corrupt the normal functioning of other components")
args = parser.parse_args()

all_data = dnn.read_bin_class_data(args.dir)

im_mean = [0.0, 0.0, 0.0]
im_std = [0.0, 0.0, 0.0]

samples = len(all_data)
print(samples)

for data in all_data:

    img = cv2.imread(args.dir + data[0], -1)
    print(img.shape)
    for i in range(3):
        im_mean[i] += np.mean(img[:,:,i])
        im_std[i] += np.std(img[:,:,i])
        
im_mean = [i/samples for i in im_mean]
im_std = [i/samples for i in im_std]

print("Results:")
print("Image: Mean: ", im_mean, "\t std: ", im_std) 

statistics_file = args.dir + args.output
with open(statistics_file, 'w') as f:
    # Im_mean[0, 1, 2], im_std[0, 1, 2], r_m, r_std, c_m, c_std, w_m, w_std, h_m, h_std
    stats = "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f" %(im_mean[0], im_mean[1], im_mean[2], im_std[0], im_std[1], im_std[2])
    f.write(stats)
