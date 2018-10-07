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
from torch.utils.data import Dataset, DataLoader

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


def read_datasets(dir):
    '''
    Read all available txt files to generate one big list
    '''
    data = []
    
    for file in os.listdir(dir):
        if (file.endswith(".csv")):
            data += read_dataset(dir + file)

    return data

def read_dataset(csv_file):
    
    data = []
    with open(csv_file, 'r') as f:
        entries = csv.reader(f, delimiter=',')
        
        for entry in entries:
            data.append([entry[0], int(entry[1]), int(entry[2]), int(entry[3]), int(entry[4]), int(entry[5])])

    return data
    
def train(args, network, optimizer, loader):

    start = time.time()
    criterion = nn.SmoothL1Loss(size_average=False)
    
    for iter in range(1, args.epochs):
        
        epoch_loss = 0
        for batch_idx, sample in enumerate(loader):
            
            images, targets = sample['image'].to(device), sample['targets'].to(device)
            optimizer.zero_grad()

            outputs = network(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        print('%s (%d %.2f%%) %.4f' % (timeSince(start, iter / args.epochs), iter, (iter / args.epochs)*100.0, epoch_loss))

        if iter % args.saverate == 0:
            save_model(network, optimizer, iter, args)        

def denormalize(output, stats):
    values = [output.data[0][i].item() for i in range(5)]
    values[1] = int(values[1]*stats["r_std"])
    values[2] = int(values[2]*stats["c_std"])
    values[3] = int(values[3]*stats["w_std"])
    values[4] = int(values[4]*stats["h_std"])
    return values

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

def show_image(image, bbox, window="image"):
    cv2.rectangle(image, (bbox[2], bbox[1]), (bbox[2] + bbox[3], bbox[1] + bbox[4]), (0, 255, 0), 2)
    cv2.imshow(window, image)
    cv2.waitKey(0)

class Normalize(object):
    def __init__(self, stats):
        self.stats = stats

    def __call__(self, sample):

        image, targets = sample['image'], sample['targets']

        norm_image = ( np.asarray(image, dtype=float) - np.asarray( self.stats["img_mean"], dtype=float).reshape(1, 1, 3) ) / np.asarray(self.stats["img_std"], dtype=float).reshape(1, 1, 3)

        '''
        # Obsoleted: Output normalization causes problems in scenarios where there is no bounding box.
        r = (entry[2] - stats["r_mean"])*res_factor / stats["r_std"]
        c = (entry[3] - stats["c_mean"])*res_factor / stats["c_std"]
        w = (entry[4] - stats["w_mean"])*res_factor / stats["w_std"]
        h = (entry[5] - stats["h_mean"])*res_factor / stats["h_std"]
        '''

        # Obsoleted: Output normalization causes problems in scenarios where there is no bounding box.
        r = targets[1] / self.stats["r_std"]
        c = targets[2] / self.stats["c_std"]
        w = targets[3] / self.stats["w_std"]
        h = targets[4] / self.stats["h_std"]
        
        return {'image':norm_image, 'targets':[targets[0], r, c, w, h]}

class Scale(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        image, targets = sample['image'], sample['targets']
        
        if self.factor > 1.0:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA

        image = cv2.resize(image, None, fx=self.factor, fy=self.factor, interpolation=interpolation)
        r, c, w, h = [int(targets[i]*self.factor) for i in range(1, 5)]
        targets = [targets[0], r, c, w, h]
        
        return {'image':image, 'targets':targets}

class ToTensor(object):

    def __call__(self, sample):
        image, targets = sample['image'], sample['targets']
        image = image.transpose((2, 0, 1))

        return {'image': torch.tensor(image, dtype=torch.float), 'targets': torch.tensor(targets, dtype=torch.float)}
    
class DroneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.drone_data = read_datasets(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.drone_data)

    def __getitem__(self, idx):
        image, targets = cv2.imread(self.root_dir + self.drone_data[idx][0], -1), self.drone_data[idx][1:]

        sample = {'image': image, 'targets': targets}

        if self.transform:
            sample = self.transform(sample)

        return sample

