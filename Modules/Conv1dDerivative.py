import torch
import torch.nn as nn
import tensorflow as tf

class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, in_channels=1, out_channels=1, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()
        self.resol = resol  # Adjust for the resolution of the temporal difference
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        #tf.print("self.in_channels", self.in_channels, "self.out_channels",self.out_channels)
        # Create a single convolution filter with appropriate padding to maintain temporal dimension
        self.filter = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, 
                                padding=(kernel_size // 2), bias=False)  # Ensure padding is set correctly

        # Assign the finite difference filter as a fixed parameter
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        # Apply the convolution operation, output should have the same temporal length as the input
        derivative = self.filter(input)
        return derivative / self.resol
