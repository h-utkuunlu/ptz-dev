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
parser.add_argument('basename', help='Base name used to identify the particular training trial')
parser.add_argument('-d', '--dataloc', default="datasets/", help='Path to the datasets')
parser.add_argument('-S', '--save_dir', default="saved_models/", help='Path to save trained models')
parser.add_argument('-o', '--open', default=None, help='Open the specified model for loading and further training')
parser.add_argument('-b', '--batch', default=5, help='Specified batch size for training', type=int) 
parser.add_argument('-e', '--epochs', default=30, help='Number of epochs to train', type=int)
parser.add_argument('-l', '--lr', default=1e-3, help='Learning rate set for the optimizer', type=float)
parser.add_argument('-L', '--logdir', default="training_logs/", help="Folder where the training logs are saved")
parser.add_argument('-r', '--saverate', default=5, help='The interval to save model states', type=int)
parser.add_argument('-w', '--workers', default=4, help='Number of workers for batch processing', type=int)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_stats = dnn.read_stats(args.dataloc + "train/stats")
test_stats = dnn.read_stats(args.dataloc + "test/stats")

train_dataset = dnn.DroneDataset(root_dir=args.dataloc + "train/", transform=transforms.Compose([ dnn.RandomCrop(224), dnn.Corrupt(0.4), dnn.Normalize(training_stats), dnn.ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

test_dataset = dnn.DroneDataset(root_dir=args.dataloc + "test/", transform=transforms.Compose([ dnn.RandomCrop(224), dnn.Normalize(test_stats), dnn.ToTensor()]))
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

network = models.resnet152(pretrained=True)

#for param in network.parameters():
#    param.requires_grad = False

num_ftrs = network.fc.in_features
network.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
network = network.to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
criterion = nn.BCELoss(size_average=False)

print("Model created")

model_name = args.open
if model_name is not None:
    dnn.load_model(model_name, network, optimizer)

## Create log file
timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
logname = args.basename + "_" + timestamp + ".txt"
params = []
params.append("Basename: " + args.basename)
params.append("Data loc: " + args.dataloc)
params.append("Opened model name: " + str(args.open))
params.append("Learning rate: " + str(args.lr))
params.append("Batch size: " + str(args.batch))
params.append("Num. of Epochs: " + str(args.epochs))

with open(args.logdir + logname, 'w') as f:
    for i in params:
        print(i)
        f.write(i + "\n")

## Start Training
for iter in range(1, args.epochs+1):

    start = time.time()
    epoch_loss = dnn.train_epoch(network, criterion, optimizer, train_loader)
    metrics = dnn.evaluate(network, test_loader)

    out_string = '%s (%d %.2f%%) Loss: %.4f Acc: %.3f Prc: %.3f Rec: %.3f' % (dnn.timeSince(start, iter / args.epochs), iter, (iter / args.epochs)*100.0, epoch_loss, metrics['accuracy'], metrics['precision'], metrics['recall'])
    print(out_string)

    with open(args.logdir + logname, 'a') as f:
        f.write(out_string + "\n")
    
    if iter % args.saverate == 0:
        dnn.save_model(network, optimizer, iter, args)        



