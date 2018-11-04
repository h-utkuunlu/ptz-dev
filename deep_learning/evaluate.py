import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
import model
import dnn_functions as dnn
import argparse
import os
import time

def generate_testlist(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(".jpeg") or file.endswith("jpg"):
            images.append(file)
    return images

parser = argparse.ArgumentParser()
parser.add_argument('open', help='Open the specified model for evaluation')
#parser.add_argument('test', help='Path to the folder to evaluate images')
parser.add_argument('-d', '--dataloc', default="datasets/", help='Path to the datasets')
parser.add_argument('-b', '--batch', default=1, help='Specified batch size for testing', type=int) 
parser.add_argument('-s', '--stats', default='stats', help='.csv file that stores the average and mean values required for normalization process')
parser.add_argument('-w', '--workers', default=1, help='Number of workers for batch processing', type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_stats = dnn.read_stats(args.dataloc + "test/stats")
test_dataset = dnn.DroneDataset(root_dir=args.dataloc + "test/", transform=transforms.Compose([ dnn.RandomCrop(224), dnn.Normalize(test_stats), dnn.ToTensor()]))
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

# Pretrained
network = models.resnet152(pretrained=True)
for param in network.parameters():
    param.requires_grad = False

num_ftrs = network.fc.in_features
network.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
network = network.to(device)

print("Model created")
#dnn.load_model(args.open, network)

## Start Evaluating
acc = dnn.timed_evaluate(network, test_loader, gui=True)
print("Overall accuracy:", acc)
