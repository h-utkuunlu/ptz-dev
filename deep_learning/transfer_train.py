import torch
import torch.nn as nn
from torch.autograd import Variable
import model
import dnn_functions as dnn
import argparse
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataloc', default="datasets/", help='Path to the datasets')
parser.add_argument('-S', '--save_dir', default="saved_models/", help='Path to save trained models')
parser.add_argument('-o', '--open', default=None, help='Open the specified model for loading and further training')
parser.add_argument('-b', '--batch', default=5, help='Specified batch size for training', type=int) 
parser.add_argument('-e', '--epochs', default=30, help='Number of epochs to train', type=int)
parser.add_argument('-l', '--lr', default=1e-4, help='Learning rate set for the optimizer', type=float)
parser.add_argument('-s', '--stats', default='stats', help='Path to file that stores the average and mean values required for normalization process')
parser.add_argument('-n', '--basename', default='net', help='Base name used to identify the particular training set')
parser.add_argument('-r', '--saverate', default=5, help='The interval to save model states', type=int)
parser.add_argument('-w', '--workers', default=4, help='Number of workers for batch processing', type=int)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = dnn.read_stats(args.dataloc + args.stats)

dataset = dnn.DroneDataset(root_dir=args.dataloc, transform=transforms.Compose([ dnn.Scale(0.44444), dnn.Normalize(stats), dnn.Squash(224), dnn.ToTensor()]))
loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

network = models.resnet101(pretrained=True)

for param in network.parameters():
    param.requires_grad = False

num_ftrs = network.fc.in_features
network.fc = nn.Linear(num_ftrs, 5)
network = network.to(device)

optimizer = torch.optim.Adadelta(network.fc.parameters(), lr=args.lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3)

print("Model created")

model_name = args.open
if model_name is not None:
    dnn.load_model(model_name, network, optimizer)

## Start Training
dnn.train(args, network, optimizer, loader)

