import torch
import torch.nn as nn
#torch.nn.utils.parametrizations.weight_norm(module, name='weight', dim=0)

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

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

import Modules.MultiHeadSpatialAttention 
reload(Modules.MultiHeadSpatialAttention)  # mandatory to reload content at each re-call atfer modification
from Modules.MultiHeadSpatialAttention import *



##############################################################
class decoder_block(nn.Module):
    ''' Decoder block with transposed CNN '''
    def __init__(self,input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding,dropout_prob=0.5):
        
        super(decoder_block, self).__init__()
        
        self.input_channels = hidden_channels 
        self.hidden_channels = input_channels
        self.output_kernel_size = input_kernel_size  
        self.output_stride = input_stride
        self.output_padding = input_padding

                
        self.deconv = nn.ConvTranspose2d(
            in_channels=self.input_channels,  # Number of input channels
            out_channels=self.hidden_channels,  # Number of output channels
            kernel_size=self.output_kernel_size,  # Size of the transposed convolutional kernel (e.g., 3x3)
            stride=self.output_stride,  # Stride of the transposed convolution operation (e.g., (2, 2))
            padding=input_padding,  # Padding added to the output tensor
            output_padding=self.output_padding,  # Additional size added to the output to adjust for the convolution operation
            bias=True  # Whether to include bias terms in the transposed convolution
        )
        self.deconv =weight_norm(self.deconv )#
        self.activation = nn.ReLU()  
        self.dropout = nn.Dropout(dropout_prob)
        #self.batch_norm = nn.BatchNorm2d(input_channels)
        
        nn.init.xavier_uniform_(self.deconv.weight)
        
    def forward(self, x):
        #tf.print("x in decoder", x.shape)

        x = x.to(self.deconv.bias.dtype)
        
        x = self.deconv(x)
        #x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
