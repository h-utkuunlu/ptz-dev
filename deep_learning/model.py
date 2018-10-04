import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

printstate = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv5 = nn.Conv2d(32, 48, 5, padding=2)
        self.conv6 = nn.Conv2d(48, 48, 5, padding=2)
        self.conv7 = nn.Conv2d(48, 64, 5, padding=2)
        self.conv8 = nn.Conv2d(64, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(3, 3)

        self.fc1 = nn.Linear(3200, 1024)
        self.fc2 = nn.Linear(1024, 100)
        self.fc3 = nn.Linear(100, 5)

        self.sigmoid = nn.Sigmoid()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.zero = torch.zeros([1, 5], dtype=torch.float).to(self.device)
        self.zero[0][0] = 1.0
        self.one = torch.ones([1, 5], dtype=torch.float).to(self.device)

        self.out1 = []
       
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        if printstate:
            print(out.size())
        out = self.pool(out)
        if printstate:
            print(out.size())
            
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        if printstate:
            print(out.size())
        out = self.pool(out)
        if printstate:
            print(out.size())

        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        if printstate:
            print(out.size())
        out = self.pool(out)
        if printstate:
            print(out.size())

        out = F.relu(self.conv7(out))
        out = F.relu(self.conv8(out))
        if printstate:
            print(out.size())
        out = self.pool(out)
        if printstate:
            print(out.size())
        
        self.out1 = out
        out = out.view(out.size(0), -1)
        if printstate:
            print(out.size())

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        probs = self.sigmoid(out.permute(1, 0)[0])
        #print(probs)

        masked_out = torch.FloatTensor(0).to(self.device)
        
        for iter, i in enumerate(probs):
            if i < 0.5:
                masked_out = torch.cat((masked_out, torch.mul(out[iter], self.zero)), 0)
            else:
                masked_out = torch.cat((masked_out, torch.mul(out[iter], self.one)), 0)

        #print(masked_out)
        return masked_out



