import random
import cv2
import numpy as np
from time import time
import math
import csv
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
def initialize_net(model_path):
    network = models.resnet50(pretrained=True)
    for param in network.parameters():
       param.requires_grad = False

    num_ftrs = network.fc.in_features
    network.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    network = network.to(device)

    if model_path is not None:
        load_model(model_path, network)
        
    network.eval()
        
    # Run random sequences to initialize
    placeholder = torch.rand((400, 3, 224, 224)).to(device)
    for i in range(5):
        start = time()
        _ = network(placeholder)
        #print(time() - start)

    return network

def read_stats(file):
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for raw_stats in reader:
            stats = {
                "img_mean": [float(raw_stats[0]), float(raw_stats[1]), float(raw_stats[2])],
                "img_std": [float(raw_stats[3]), float(raw_stats[4]), float(raw_stats[5])]
            }
    return stats

class PrepareRTImage(object):
        
    def __init__(self, size, stats):
        self.size = size
        self.stats = stats

    def normalize(self, image):
        norm_image = ( np.asarray(image, dtype=float) - np.asarray(self.stats["img_mean"], dtype=float).reshape(1, 1, 3) ) / np.asarray(self.stats["img_std"], dtype=float).reshape(1, 1, 3)
        return image

    def toTensor(self, data):
        data = [image.transpose((2, 0, 1)) for image in data] # Channel before x-y
        return torch.tensor(data, dtype=torch.float)

    def resize(self, image, min_dim):
        h, w, _ = image.shape

        if h != w: # Erroneous bbox adjustment
            print("Image dimensions are not the same!")
        
        if float(self.size)/float(min(h, w)) != 1:
            if min(h, w) < self.size:
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA
        else:
            return image
        
        image = cv2.resize(image, (self.size, self.size), interpolation=interpolation)
        return image
    
    def __call__(self, images):
        data = [self.normalize(self.resize(image, self.size)) for image in images]        
        return self.toTensor(data)

def real_time_evaluate(network, data):

    with torch.no_grad():
        images = data.to(device)
        outputs = network(images)
        results = (outputs.cpu().numpy() > 0.5).astype(int)
        
    return results

def load_model(model_name, network, optimizer=None):

    state = torch.load(model_name)
    new_state = deepcopy(state)
    
    for i in state['network_state'].keys():
        new_state['network_state'][i.replace('module.', '', 1)] = new_state['network_state'].pop(i)

    network.load_state_dict(new_state['network_state'])
    if optimizer is not None:
        optimizer.load_state_dict(new_state['optimizer_state'])
    state = None
    
    print("Model load successful")
