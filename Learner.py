import torch
from torch import nn


class Learner(nn.Module):
    def __init__(self, in_dim, num_class):
        super(Learner, self).__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_dim, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.out = nn.Linear(100, num_class)
        self.batch_norm = nn.BatchNorm1d(in_dim)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.batch_norm(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        
        return x