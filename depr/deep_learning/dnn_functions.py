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

def read_bin_class_data(dir):

    positive = []
    negative = []

    for file in os.listdir(dir + "positive/"):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".JPEG"):
            positive.append(["positive/" + file, 1])
        
    for file in os.listdir(dir + "negative/"):
        if file.endswith(".jpg") or file.endswith(".JPEG"):
            negative.append(["negative/" + file, 0])
    
    return positive + negative
        
def read_stats(file):
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for raw_stats in reader:
            stats = {
                "img_mean": [float(raw_stats[0]), float(raw_stats[1]), float(raw_stats[2])],
                "img_std": [float(raw_stats[3]), float(raw_stats[4]), float(raw_stats[5])]
            }
            '''
            # Obsolete: No box regression
            "r_mean": float(raw_stats[6]),
            "r_std": float(raw_stats[7]),
            "c_mean": float(raw_stats[8]),
            "c_std": float(raw_stats[9]),
            "w_mean": float(raw_stats[10]),
            "w_std": float(raw_stats[11]),
            "h_mean": float(raw_stats[12]),
            "h_std": float(raw_stats[13])
            '''
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
    
def train_epoch(network, criterion, optimizer, loader):

    epoch_loss = 0
    for batch_idx, sample in enumerate(loader):
            
        images, targets = sample['image'].to(device), sample['targets'].unsqueeze(1).to(device)
        optimizer.zero_grad()

        outputs = network(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss

def train(args, network, optimizer, loader):

    start = time.time()
    criterion = nn.BCELoss(size_average=False)
    network.train()
    
    for iter in range(1, args.epochs+1):
        
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

def evaluate(network, loader):

    total = 0
    correct = 0
    actual_drones = 0
    predicted_drones = 0
    true_pos = 0
    
    with torch.no_grad():
        network.eval()
        
        for i, sample in enumerate(loader):
            images, targets = sample['image'].to(device), sample['targets'].unsqueeze(1).to(device)

            threshold = torch.Tensor([0.5]).to(device)
            outputs = network(images)
            predictions = outputs > threshold

            total += targets.size(0)
            correct += (predictions == targets.byte()).sum().item()
            actual_drones += targets.sum().item() # for recall
            predicted_drones += predictions.sum().item()
            true_pos += (predictions & targets.byte()).sum().item()

    accuracy = 100.0 * (correct / total)
    precision = 100.0 * (true_pos / predicted_drones)
    recall = 100.0 * (true_pos / actual_drones)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}

def timed_evaluate(network, loader, gui=False):

    durations = []
    total = 0
    correct = 0
    
    if gui:
        cv2.namedWindow("show_batch")
    
    with torch.no_grad():
        network.eval()
        
        for i, sample in enumerate(loader):
            images, targets = sample['image'].to(device), sample['targets'].unsqueeze(1).to(device)
            batch_size = images.size(0)
            total += targets.size(0)
            
            threshold = torch.Tensor([0.5]).to(device)

            start = time.time()
            outputs = network(images)
            predictions = outputs > threshold

            duration = time.time() - start
            durations.append(duration)

            correct += (predictions == targets.byte()).sum().item()
            
            #for i in range(batch_size):
                
            #    img = images[i].permute(1, 2, 0).numpy()
            #    print(img.shape)

            
            hcatim = [images[i].permute(1, 2, 0).numpy() for i in range(batch_size)]
            hcatim = np.hstack(hcatim)
            print(hcatim.shape)
            #exit()

            print("Duration: ", duration)
            print(predictions)

            if gui:
                cv2.imshow("show_batch", hcatim)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
    
    accuracy = (correct / total) * 100.0
            
    return {'accuracy': accuracy}


def denormalize(output, stats):
    values = [output.data[0][i].item() for i in range(5)]
    values[1] = int(values[1]*stats["r_std"])
    values[2] = int(values[2]*stats["c_std"])
    values[3] = int(values[3]*stats["w_std"])
    values[4] = int(values[4]*stats["h_std"])
    return values

def load_model(model_name, network, optimizer=None):

    state = torch.load(model_name)

    network.load_state_dict(state['network_state'])
    if optimizer is not None:
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
        
        # Obsoleted: Classification problem instead of regression
        r = targets[1] / self.stats["r_std"]
        c = targets[2] / self.stats["c_std"]
        w = targets[3] / self.stats["w_std"]
        h = targets[4] / self.stats["h_std"]
        '''
        
        return {'image': norm_image, 'targets': targets}

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

# Obsolete: Eliminated squashing
class Squash(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, targets = sample['image'], sample['targets']

        res_im = cv2.resize(image, (224, 224))

        return {'image': res_im, 'targets':targets}

class RandomCrop(object):
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        image, targets = sample['image'], sample['targets']
        h, w = image.shape[:2]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top+new_h, left:left+new_h]

        return {'image': image, 'targets':targets}
        
class Corrupt(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        image, targets = sample['image'], sample['targets']
        if np.random.random() > self.prob:
            return {'image': image, 'targets': targets}

        else:
            corruptions = ["agbn", "blur", "occlusion"] # agbn: additive gaussian + bias
            chosen = random.choice(corruptions)
            if chosen == "agbn":
                image = self.gaussianNoise(self.biasNoise(image))
            elif chosen == "blur":
                image = self.blur(image)
            elif chosen == "occlusion":
                image = self.occlude(image)

            return {'image': image, 'targets': targets}
            
    @staticmethod
    def gaussianNoise(image):
        row, col, ch = image.shape
        mean = 0
        var = 0.0025
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))*255
        gauss = np.repeat(gauss.reshape(row, col, 1), 3, axis=2)
        noisy = np.clip(image.astype(np.int32) + gauss.astype(np.int32), 0, 255).astype(np.uint8)
        
        return noisy

    @staticmethod
    def biasNoise(image):
        row, col, ch = image.shape
        val = (random.random()-0.5)*(int(np.amax(image)) + int(np.amin(image)))*0.25
        mask = np.ones((row, col, ch), dtype=np.float64)*val
        noisy = np.clip(image.astype(np.int32) + mask.astype(np.int32), 0, 255).astype(np.uint8)
    
        return noisy

    @staticmethod
    def blur(image):
        return cv2.GaussianBlur(image, (27, 27), 0)

    @staticmethod
    def occlude(image):
        row, col, ch = image.shape
        w = col // 3
        h = row // 3
        occlusion = np.zeros((h, w, ch))
        y = np.random.randint(0, high=(row - h + 1))
        x = np.random.randint(0, high=(col - w + 1))
        occluded = image.copy()
        occluded[y:y+h, x:x+w] = occlusion

        return occluded             
    
class ToTensor(object):

    def __call__(self, sample):
        image, targets = sample['image'], sample['targets']
        image = image.transpose((2, 0, 1))

        return {'image': torch.tensor(image, dtype=torch.float), 'targets': torch.tensor(targets, dtype=torch.float)}
    
class DroneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.drone_data = read_bin_class_data(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.drone_data)

    def __getitem__(self, idx):
        #print(self.root_dir + self.drone_data[idx][0])
        image, targets = cv2.imread(self.root_dir + self.drone_data[idx][0], -1), self.drone_data[idx][1]

        sample = {'image': image, 'targets': targets}

        if self.transform:
            sample = self.transform(sample)

        return sample

