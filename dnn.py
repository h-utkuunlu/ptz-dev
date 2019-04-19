"Provides support to convert, transform and classify images"
from time import time
import csv

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from fastai.vision import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_net(model_path):
    "Returns pytorch CNN."
    network = models.resnet50(pretrained=True)
    for param in network.parameters():
        param.requires_grad = False

    num_ftrs = network.fc.in_features
    network.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    network = nn.DataParallel(network).to(device)

    if model_path is not None:
        load_model(model_path, network)

    network.eval()

    # Run random sequences to initialize bigger chunk of the memory
    placeholder = torch.rand((400, 3, 224, 224)).to(device)
    for _ in range(5):
        start = time()
        _ = network(placeholder)
        #print(time() - start)

    return network


def initialize_net_fastai(model_folder_path, model_file_name='export.pkl'):
    "Returns fastai learner."
    return load_learner(model_folder_path, fname=model_file_name)


def read_stats(file):
    "Returns image stats from `file` as dictionary."
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for raw_stats in reader:
            stats = {
                "img_mean": [
                    float(raw_stats[0]),
                    float(raw_stats[1]),
                    float(raw_stats[2])
                ],
                "img_std": [
                    float(raw_stats[3]),
                    float(raw_stats[4]),
                    float(raw_stats[5])
                ]
            }
    return stats


def real_time_evaluate(network, data):
    "Returns predictions for the images provided in `data` by pytorch CNN."
    with torch.no_grad():
        images = data.to(device)
        outputs = network(images)
        probs = outputs.cpu().numpy()
        results = (probs > 0.85).astype(int)

    return results


def real_time_evaluate_fastai(network, data):
    "Returns predictions for the images provided in `data` by fastai learner."
    images = [Image(image) for image in data]
    results = [network.predict(img)[1].item()
               for img in images]  # TODO: run this in parallel

    return results


def load_model(model_name, network, optimizer=None):
    "Loads model at `model_name` into `network` (optional optimizer change)."
    state = torch.load(model_name)

    network.load_state_dict(state['network_state'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state'])
    state = None

    print("Model load successful")


class Normalize(object):
    "Normalizes image based on training data."

    def __init__(self, stats):
        self.stats = stats

    def __call__(self, image):
        img_mean = np.asarray(self.stats["img_mean"],
                              dtype=float).reshape(1, 1, 3)
        img_std = np.asarray(self.stats["img_std"],
                             dtype=float).reshape(1, 1, 3)
        norm_image = (np.asarray(image, dtype=float) - img_mean) / img_std

        return norm_image


class Resize(object):
    "Resizes images to square according to `size` attribute."

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if self.size > sample.shape[0]:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA

        image = cv2.resize(sample, (self.size, self.size),
                           interpolation=interpolation)

        return image


class ToTensor(object):
    "Converts image to tensor."

    def __call__(self, sample):
        image = sample.transpose((2, 0, 1))
        return torch.tensor(image, dtype=torch.float).unsqueeze(0)
