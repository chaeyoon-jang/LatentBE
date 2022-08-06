from tkinter import X
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import math

class Conv2DBatchEnsemble(nn.Module):
    
    def __init__ (self, ensemble_size, in_channels, out_channels, kernel_size, padding=0, bias_is=False, inference=False):
        super(Conv2DBatchEnsemble, self).__init__()
        self.inference = inference
        self.ensemble_size = ensemble_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
        )
        self.r_factor = nn.Parameter(torch.Tensor(self.ensemble_size, self.in_channels))
        self.s_factor = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_channels))
        
        self.bias_is = bias_is
        if self.bias_is:
            self.bias = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.conv.weight, mean=1.0, std=0.1)
        nn.init.ones_(self.r_factor)
        nn.init.ones_(self.s_factor) 
        
        if self.bias_is:
            nn.init.zeros_(self.bias)
        
    def forward(self, x):
        
        if self.inference:
            out = self.conv(x)
            _, c_out, h_out, w_out = out.size()
            out = out + self.bias.view(1, c_out, 1, 1)
            return out 
        
        else:
            # X : [batch_size, in_channels, height, weight]
            _, c_in, h_in, w_in = x.size()
            # X : [ensemble_size, batch_size/ensemble_size, in_channels, height, weight]
            x = x.view(self.ensemble_size, -1, c_in, h_in, w_in)
            # X * R : [ensemble_size, batch_size/ensemble_size, in_channels, height, weight]
            rx = x.mul(self.r_factor.view(self.ensemble_size, 1, c_in, 1, 1))
            # X * R : [batch_size, in_channels, height, weight]
            rx = rx.view(-1, c_in, h_in, w_in)
            # (X * R)*W : [batch_size, out_channels, height, weight]
            wrx = self.conv(rx)
            _, c_out, h_out, w_out = wrx.size()
            # (X * R)*W : [ensemble_size, batch_size/ensemble_size, out_channels, heigth, weight]
            y = wrx.view(self.ensemble_size, -1, c_out, h_out, w_out)
            # ((X * R)*W)*S :[ensemble_size, batch_size/ensemble_size, out_channels, height, weight]
            y = y.mul(self.s_factor.view(self.ensemble_size, 1, c_out, 1, 1))
            # ((X * R)*W)*S + bias
            if self.bias_is:
                y = y + self.bias.view(self.ensemble_size, 1, c_out, 1, 1)
                
            y = y.view(-1, c_out, h_out, w_out)
            return y

class LinearBatchEnsemble(nn.Module):
    
    def __init__(self, ensemble_size, in_channels, out_channels, bias_is=False, inference=False):
        super(LinearBatchEnsemble, self).__init__()
        self.inference = inference
        self.ensemble_size = ensemble_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(
            self.in_channels, self.out_channels, bias=False
        )
        self.r_factor = nn.Parameter(torch.Tensor(self.ensemble_size, self.in_channels))
        self.s_factor = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_channels))
        self.bias_is = bias_is
        
        if self.bias_is:
            self.bias = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_channels))
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, mean=1.0, std=0.1)
        nn.init.ones_(self.r_factor)
        nn.init.ones_(self.s_factor) 
        
        if self.bias_is:
            nn.init.zeros_(self.bias)
        
    def forward(self, x):
        
        if self.inference:
            out = self.linear(x)
            _, fout = out.size()
            out = out + self.bias.view(1, fout)
            return out
            
        else:
            # X : [batch_size, in_channels]
            _, fout = x.size()
            # X : [ensemble_size, batch_size/ensemble_size, in_channels]
            rx = x.view(self.ensemble_size, -1, fout)
            # R * X : [ensemble_size, batch_size/ensemble_size, in_channels]
            rx = rx * self.r_factor.view(self.ensemble_size, 1, fout)
            # R * X : [batch_size, in_channels]
            rx = rx.view(-1, fout)
            # (R * X) * W : [batch_size, out_channels]
            wrx = self.linear(rx)
            _, lout = wrx.size()
            # (R * X) * W : [ensemble_size, batch_size/ensemble_size, out_channels]
            y = wrx.view(self.ensemble_size, -1, lout)
            # ((R * X) * W) * S : [ensemble_size, batch_size/ensemble_size, out_channels]
            y = y * self.s_factor.view(self.ensemble_size, 1, lout)
            # ((R * X) * W) * S + bias : [ensemble_size, batch_size/ensemble_size, out_channels]
            if self.bias_is:
                y = y + self.bias.view(self.ensemble_size, 1, lout)
            # ((R * X) * W) * S + bias = Y(final) : [batch_size, out_channels]                
            y = y.view(-1, lout)
            return y


class CNN_be(nn.Module):
    
    def __init__(self, bias_is=False, inference=False):
        super(CNN_be, self).__init__()
        
        self.layer1 = nn.Sequential(
            Conv2DBatchEnsemble(ensemble_size=4, in_channels=1, out_channels=6, kernel_size=5, bias_is=bias_is, inference=inference),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            Conv2DBatchEnsemble(ensemble_size=4, in_channels=6, out_channels=12, kernel_size=5, bias_is=bias_is, inference=inference),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = LinearBatchEnsemble(ensemble_size=4, in_channels=5*5*12, out_channels=120, bias_is=bias_is, inference=inference)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = LinearBatchEnsemble(ensemble_size=4, in_channels=120, out_channels=60, bias_is=bias_is, inference=inference)
        self.fc3 = LinearBatchEnsemble(ensemble_size=4, in_channels=60, out_channels=10, bias_is=bias_is, inference=inference)

    def forward(self, x):
            
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out