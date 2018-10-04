import torch
import torch.nn as nn
from torch.autograd import Variable
import model
import dnn_functions as dnn
import argparse
import os
import time

def generate_testlist(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(".jpeg") or file.endswith("jpg"):
            images.append(path + "/" + file)
    return images

parser = argparse.ArgumentParser()
parser.add_argument('open', help='Open the specified model for evaluation')
parser.add_argument('test', help='Path to the folder to evaluate images')
parser.add_argument('-s', '--stats', default='stats.csv', help='.csv file that stores the average and mean values required for normalization process')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = dnn.read_stats(args.stats)

network = model.Net().to(device)
optimizer = torch.optim.Adadelta(network.parameters(), lr=0)

print("Model created")

dnn.load_model(network, optimizer, args.open)

## Start Evaluating
test_images = generate_testlist(args.test)

for image in test_images:
    start = time.time()
    output = dnn.evaluate(network, image, stats)
    print(image, output)
    print(time.time() - start)

    # TODO: Display the bbox prediction on the image
