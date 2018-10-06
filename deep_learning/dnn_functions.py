import random
import cv2
import numpy as np
import time
import math
import csv
import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_stats(file):
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for raw_stats in reader:
            stats = {
                "img_mean": [float(raw_stats[0]), float(raw_stats[1]), float(raw_stats[2])],
                "img_std": [float(raw_stats[3]), float(raw_stats[4]), float(raw_stats[5])],
                "r_mean": float(raw_stats[6]),
                "r_std": float(raw_stats[7]),
                "c_mean": float(raw_stats[8]),
                "c_std": float(raw_stats[9]),
                "w_mean": float(raw_stats[10]),
                "w_std": float(raw_stats[11]),
                "h_mean": float(raw_stats[12]),
                "h_std": float(raw_stats[13])
            }

    return stats

#stats = read_stats(args.stats)

def read_datasets(dir="datasets/", statsfile="stats.csv"):
    '''
    Read all available txt files to generate one big list
    '''
    data = []
    
    for file in os.listdir(dir):
        if (file.endswith(".csv") and file != (statsfile)):
            data += read_dataset(dir + file)

    return data

def read_dataset(csv_file):
    
    data = []
    with open(csv_file, 'r') as f:
        entries = csv.reader(f, delimiter=',')
        
        for entry in entries:
            data.append([entry[0], int(entry[1]), int(entry[2]), int(entry[3]), int(entry[4]), int(entry[5])])

    return data

def read_data(entry, stats, dir="datasets/"):
    '''
    Given an entry from the csv files that describe the dataset and the statistics about all the data, outputs the workable images and entries
    entry: list that contains path to image [0], presence of a drone [1], bbox r, c, w, and h [2, 3, 4, 5]
    stats: dictionary that contains mean and std. var. for images and bbox parameters for all datasets
    '''

    #TEMPORARY: RESIZING FACTOR
    res_factor = 0.44444445

    raw_image = cv2.imread(dir + entry[0], -1)
    res_image = cv2.resize(raw_image, None, fx=res_factor, fy=res_factor, interpolation=cv2.INTER_AREA)
    image = ( np.asarray(res_image, dtype=float) - np.asarray( stats["img_mean"], dtype=float).reshape(1, 1, 3) ) / np.asarray(stats["img_std"], dtype=float).reshape(1, 1, 3)

    '''
    # Obsoleted: Output normalization causes problems in scenarios where there is no bounding box.
    r = (entry[2] - stats["r_mean"])*res_factor / stats["r_std"]
    c = (entry[3] - stats["c_mean"])*res_factor / stats["c_std"]
    w = (entry[4] - stats["w_mean"])*res_factor / stats["w_std"]
    h = (entry[5] - stats["h_mean"])*res_factor / stats["h_std"]
    '''

    # Obsoleted: Output normalization causes problems in scenarios where there is no bounding box.
    r = (entry[2])*res_factor / stats["r_std"]
    c = (entry[3])*res_factor / stats["c_std"]
    w = (entry[4])*res_factor / stats["w_std"]
    h = (entry[5])*res_factor / stats["h_std"]
   
    target = [entry[1], r, c, w, h]    
    return image, target

def load_data(entries, stats):
    '''
    Given a list of entries from data (read from csv using read_datasets), loads the images into lists, and converts them to tensor
    '''
    images, targets = [], []
    
    for entry in entries:
        image, target = read_data(entry, stats)
        images.append(image)
        targets.append(target)

    input_tensor = torch.tensor(images, dtype=torch.float, device=device).permute(0, 3, 1, 2)
    target_tensor = torch.tensor(targets, dtype=torch.float, device=device)
    
    return input_tensor, target_tensor
    
def train_model(input_tensor, target_tensor, network, optimizer, criterion):
    '''
    Train the model with one batch of images supplied as input tensors, and compute losses
    input_tensor: 4D tensor containing the images
    target_tensor: 
    ...

    '''
    batch_size = target_tensor.size(0)
    
    optimizer.zero_grad()

    #print(target_tensor)
    
    output = network(input_tensor)
    loss = criterion(output, target_tensor)

    loss.backward()
    optimizer.step()

    #print("Trained one iter.")
    
    return loss.item() / batch_size
    
def train_epochs(network, optimizer, args, stats):
    start = time.time()
    batch_size = args.batch
    criterion = nn.SmoothL1Loss(size_average=False)
    
    data = read_datasets(args.dataloc)
    
    for iter in range(1, args.epochs + 1):

        epoch_data = random.sample(data, len(data))
        epoch_loss = 0
        num_batches = len(data) // batch_size
        
        for i in range(num_batches):
            input_tensor, target_tensor = load_data(epoch_data[i*batch_size:(i+1)*batch_size], stats)
            loss = train_model(input_tensor, target_tensor, network, optimizer, criterion)
            epoch_loss += loss
        
        print('%s (%d %.2f%%) %.4f' % (timeSince(start, iter / args.epochs), iter, (iter / args.epochs)*100.0, epoch_loss))

        if iter % args.saverate == 0:
            save_model(network, optimizer, iter, args)

'''
# Obsolete: No target scaling
         
def denormalize(output, stats):
    values = [output.data[0][i].item() for i in range(5)]
    values[1] = int(values[1]*stats["r_std"] + stats["r_mean"])
    values[2] = int(values[2]*stats["c_std"] + stats["c_mean"])
    values[3] = int(values[3]*stats["w_std"] + stats["w_mean"])
    values[4] = int(values[4]*stats["h_std"] + stats["h_mean"])

    return values

'''

def evaluate(network, image_path, stats):
    with torch.no_grad():
        entry = [image_path, 0, 0, 0, 0, 0]
        input_tensor, _ =  load_data([entry], stats)
        output = network(input_tensor)
        #print(output.size())
        bbox_output = [output.data[0][i].item() for i in range(output.size(1))]
                       
        return bbox_output

def load_model(network, optimizer, model_name):

    state = torch.load(model_name)
    network.load_state_dict(state['network_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    state = None
    
    print("Model load successful")

def save_model(network, optimizer, epochs, args):
    state = {
        'epochs:': epochs,
        'learning_rate:': args.lr,
        'network_state': network.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'base': args.open
    }

    file_name = args.basename + "_" + str(epochs) + ".pkl"
    
    torch.save(state, args.save_dir + "/" + file_name)
    print("Model saved")     

# Helper functions to keep track of time

def asMinutes(s):
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
