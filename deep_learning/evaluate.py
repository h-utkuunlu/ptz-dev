import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
val_stats = dnn.read_stats(args.dataloc + "val/stats")

val_dataset = dnn.DroneDataset(root_dir=args.dataloc + "val/", transform=transforms.Compose([ dnn.Resize(224), dnn.Normalize(val_stats), dnn.ToTensor()]))
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

network = models.resnet50(pretrained=True)

#for param in network.parameters():
#    param.requires_grad = False

num_ftrs = network.fc.in_features
network.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
network = torch.nn.DataParallel(network).to(device)

print("Model created")

model_name = args.open
if model_name is None:
    print("Enter a model name")
    exit()
    
dnn.load_model(model_name, network)

if args.batch == 1:
    results = dnn.iterative_evaluate(network, val_loader)
else:
    results = dnn.evaluate(network, val_loader)
    
print(results['ground_truth'][:25])
print(results['predictions'][:25])

accuracy = accuracy_score(results['ground_truth'], results['predictions'])
precision = precision_score(results['ground_truth'], results['predictions'])
recall = recall_score(results['ground_truth'], results['predictions'])

out_string = 'Acc: %.3f Prc: %.3f Rec: %.3f' % (accuracy, precision, recall)
print(out_string)
