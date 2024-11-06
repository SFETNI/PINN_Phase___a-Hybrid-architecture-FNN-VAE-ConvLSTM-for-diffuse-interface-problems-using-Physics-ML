import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
from importlib import reload
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.nn.utils.parametrizations import weight_norm

#import scipy.optimize
import scipy.io
from torch.optim import Adam, LBFGS
from torch.utils.data import Dataset, DataLoader

import Modules.MultiHeadSpatialAttention 
reload(Modules.MultiHeadSpatialAttention)  # mandatory to reload content at each re-call atfer modification
from Modules.MultiHeadSpatialAttention import *


os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

###########################################################
class encoder_block(nn.Module):
    ''' encoder with CNN '''
    def __init__(self,  input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding,dropout_prob=0.5):
        
        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = (nn.Conv2d(self.input_channels, 
            self.hidden_channels, self.input_kernel_size, self.input_stride, 
            self.input_padding, bias=True, padding_mode='circular'))
        self.conv =weight_norm(self.conv )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        #self.batch_norm = nn.BatchNorm2d(hidden_channels)

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        #x = self.batch_norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

