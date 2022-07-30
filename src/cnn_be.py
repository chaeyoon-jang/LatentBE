from tkinter import X
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import math

class Conv2d_BatchEnsemble(nn.Module):
    
    def __init__ (self, ensemble_size, in_channels, out_channels, kernel_size, padding=0, inference=False):
        super(Conv2d_BatchEnsemble, self).__init__()
        self.inference = inference
        self.ensemble_size = ensemble_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
        )
        self.r = nn.Parameter(torch.Tensor(self.ensemble_size, self.in_channels))
        self.s = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_channels))
        self.b = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_channels))
        self.reset_parameters()
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2))
        
    def reset_parameters(self):
        nn.init.normal_(self.r, mean=1.0, std=0.1)
        nn.init.normal_(self.s, mean=1.0, std=0.1) 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)
        
    def forward(self, x):
        
        if self.inference:
            return self.conv(x)
        
        else:
            _, c_in, h_in, w_in = x.size()
            x = x.view(self.ensemble_size, -1, c_in, h_in, w_in)
            rx = x * (1.0 + self.r.view(self.ensemble_size, 1, c_in, 1, 1))
            rx = rx.view(-1, c_in, h_in, w_in)
            wrx = self.conv(rx)
            _, c_out, h_out, w_out = wrx.size()
            y = wrx.view(self.ensemble_size, -1, c_out, h_out, w_out)
            y = y * (1.0 + self.s.view(self.ensemble_size, 1, c_out, 1, 1))
            y = y + self.b.view(self.ensemble_size, 1, c_out, 1, 1)
            y = y.view(-1, c_out, h_out, w_out)
            #print(y.size())
            return y

class Linear_BatchEnsemble(nn.Module):
    
    def __init__(self, ensemble_size, in_channels, out_channels, inference=False):
        super(Linear_BatchEnsemble, self).__init__()
        self.inference = inference
        self.ensemble_size = ensemble_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(
            self.in_channels, self.out_channels, bias=False
        )
        self.r = nn.Parameter(torch.Tensor(self.ensemble_size, self.in_channels))
        self.s = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_channels))
        self.b = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_channels))
        self.reset_parameters()
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=np.sqrt(2))
    
    def reset_parameters(self):
        nn.init.normal_(self.r, mean=1.0, std=0.1)
        nn.init.normal_(self.s, mean=1.0, std=0.1) 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)
        
    def forward(self, x):
        
        if self.inference:
            return self.linear(x)
            
        else:
            _, fout = x.size()
            rx = x.view(self.ensemble_size, -1, fout)
            rx = rx * self.r.view(self.ensemble_size, 1, fout)
            rx = rx.view(-1, fout)
            wrx = self.linear(rx)
            _, lout = wrx.size()
            y = wrx.view(self.ensemble_size, -1, lout)
            y = y * self.s.view(self.ensemble_size, 1, lout) 
            y = y + self.b.view(self.ensemble_size, 1, lout)
            y = y.view(-1, lout)
            #print(y.size())
            return y


class CNN_be(nn.Module):
    
    def __init__(self, inference=False):
        super(CNN_be, self).__init__()
        
        self.layer1 = nn.Sequential(
            Conv2d_BatchEnsemble(ensemble_size=4, in_channels=1, out_channels=6, kernel_size=5, inference=inference),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            Conv2d_BatchEnsemble(ensemble_size=4, in_channels=6, out_channels=12, kernel_size=5, inference=inference),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = Linear_BatchEnsemble(ensemble_size=4, in_channels=5*5*12, out_channels=120, inference=inference)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = Linear_BatchEnsemble(ensemble_size=4, in_channels=120, out_channels=60, inference=inference)
        self.fc3 = Linear_BatchEnsemble(ensemble_size=4, in_channels=60, out_channels=10, inference=inference)

    def forward(self, x):
            
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out