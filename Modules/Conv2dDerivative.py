import torch
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



partial_x_sobel_op = torch.tensor([[[[ -1.,   -6.,  -15.,  -20.,  -15.,   -6.,   -1.],
                                    [  -4.,  -24.,  -60.,  -80.,  -60.,  -24.,   -4.],
                                    [  -5.,  -30.,  -75., -100.,  -75.,  -30.,   -5.],
                                    [   0.,    0.,    0.,    0.,    0.,    0.,    0.],
                                    [   5.,   30.,   75.,  100.,   75.,   30.,    5.],
                                    [   4.,   24.,   60.,   80.,   60.,   24.,    4.],
                                    [   1.,    6.,   15.,   20.,   15.,    6.,    1.]]]], dtype=torch.float32)



partial_y_sobel_op = torch.tensor([[[[  -1.,   -4.,   -5.,    0.,    5.,    4.,    1.],
                                    [  -6.,  -24.,  -30.,    0.,   30.,   24.,    6.],
                                    [ -15.,  -60.,  -75.,    0.,   75.,   60.,   15.],
                                    [ -20.,  -80., -100.,    0.,  100.,   80.,   20.],
                                    [ -15.,  -60.,  -75.,    0.,   75.,   60.,   15.],
                                    [  -6.,  -24.,  -30.,    0.,   30.,   24.,    6.],
                                    [  -1.,   -4.,   -5.,    0.,    5.,    4.,    1.]]]], dtype=torch.float32)



laplace_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]] 
             
"""

partial_y_sobel_op_list = [[[[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [1/12, -8/12, 0, 8/12, -1/12],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]]]]

partial_x_sobel_op_list = [[[[0, 0, 1/12, 0, 0],
                             [0, 0, -8/12, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 8/12, 0, 0],
                             [0, 0, -1/12, 0, 0]]]]
"""

# Compute the scaling factors
scale_x = torch.sum(torch.abs(partial_x_sobel_op ))
scale_y = torch.sum(torch.abs(partial_y_sobel_op ))

# Scale the Sobel operators
partial_x_sobel_op = partial_x_sobel_op / scale_x
partial_y_sobel_op = partial_y_sobel_op / scale_y

########################################
#"""
class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, in_channels=1, out_channels=1, kernel_size=7, name=''):
        super(Conv2dDerivative, self).__init__()
        
        self.resol = resol  
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        #print( self.in_channels,  self.out_channels )

        self.filters = nn.ModuleList([
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=(self.kernel_size - 1) // 2, bias=False)
            for _ in range(self.out_channels)
        ])

        for filter in self.filters:
            filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):       
        derivatives = [filter(input[:, i:i+1, :, :]) for i, filter in enumerate(self.filters)]  
        derivative = torch.cat(derivatives, dim=1)

        return derivative / self.resol
