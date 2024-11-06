import torch
#import cv2
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.nn.utils import weight_norm
#import scipy.optimize
import scipy.io
from torch.optim import Adam, LBFGS
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)
import torch.nn.init as init

###################################################
import torch
import torch.nn as nn
import torch.nn.init as init

class ConvLSTMCell(nn.Module):
    ''' Convolutional LSTM '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
                input_stride, input_padding, dropout_prob=0.5):

        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.hidden_kernel_size = 3
        self.dropout_prob = dropout_prob

        # Define convolutional layers for gates
        self.Wxr = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True)

        self.Whr = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False)

        self.Wxz = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True)

        self.Whz = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False)

        self.Wxh = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True)

        self.Whh = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False)

        # Dropout is typically used after the ConvGRUCell
        self.dropout = nn.Dropout(p=self.dropout_prob)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        init.constant_(self.Wxr.bias, 0.0)  # Reset gate bias
        init.constant_(self.Wxz.bias, 0.0)  # Update gate bias
        init.constant_(self.Wxh.bias, 0.0)  # New hidden state bias

    def forward(self, x, h):
        # Reset gate
        rt = torch.sigmoid(self.Wxr(x) + self.Whr(h))
        
        # Update gate
        zt = torch.sigmoid(self.Wxz(x) + self.Whz(h))
        
        # New hidden state
        h_tilde = torch.tanh(self.Wxh(x) + rt * self.Whh(h))
        
        # Update hidden state
        ht = (1 - zt) * h + zt * h_tilde

        # Apply dropout
        x_out = self.dropout(ht)

        return x_out, ht
"""
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_features = 4
        self.dropout_prob = dropout_prob

        # Define convolutional layers for gates
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True)

        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False)

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True)

        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False)

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True)

        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False)

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True)

        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False)

        #self.bi = nn.Parameter(torch.zeros(self.hidden_channels, 1, 1))
        #self.bf = nn.Parameter(torch.ones(self.hidden_channels, 1, 1))
        #self.bc = nn.Parameter(torch.zeros(self.hidden_channels, 1, 1))
        #self.bo = nn.Parameter(torch.zeros(self.hidden_channels, 1, 1))

        # Dropout is typically used after the ConvLSTMCell
        #self.dropout = nn.Dropout(p=self.dropout_prob)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        init.constant_(self.Wxf.bias, 1.0)  # Forget gate bias
        init.constant_(self.Wxi.bias, 0.0)  # Input gate bias
        init.constant_(self.Wxo.bias, 0.0)  # Output gate bias
        init.constant_(self.Wxc.bias, 0.0)  # Cell gate bias

    def forward(self, x, h, c):
        #  input gate 
        it = torch.sigmoid(self.Wxi(x) + self.Whi(h) )
        
        #  forget gate
        ft = torch.sigmoid(self.Wxf(x) + self.Whf(h) )
        
        #  output gate
        ot = torch.sigmoid(self.Wxo(x) + self.Who(h) )
        
        #  new candidate cell state
        zt = torch.tanh(self.Wxc(x) + self.Whc(h) )
        
        # Update cell state
        ct = ft * c + it * zt
        
        #  hidden state
        ht = ot * torch.tanh(ct)

        x_out = ht

        return x_out, ht, ct

"""

##############################################################################
