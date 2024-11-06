import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.nn.utils import weight_norm
#import scipy.optimize
import scipy.io
from torch.optim import Adam, LBFGS
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

import Modules.encoder_block
reload(Modules.encoder_block)  # mandatory to reload content at each re-call atfer modification
from Modules.encoder_block import *

import Modules.decoder_block
reload(Modules.decoder_block)  # mandatory to reload content at each re-call atfer modification
from Modules.decoder_block import *

import Modules.decoder_block
reload(Modules.decoder_block)  # mandatory to reload content at each re-call atfer modification
from Modules.decoder_block import *
########################################
"""
The encoder block consists of a series of convolutional layers followed by ReLU activation functions.
It takes the input image tensor and transforms it into a latent representation.
Each convolutional layer extracts higher-level features from the input image, gradually reducing its spatial dimensions and increasing the number of channels.
The final output of the encoder block is a high-dimensional feature tensor representing the input image in a compressed form.
Mean and Variance Layers:

After encoding the input image into a high-dimensional feature tensor, the mean and variance layers further process this tensor to extract the mean and variance vectors of the latent space.
These layers typically consist of linear (fully connected) layers that map the high-dimensional feature tensor to the mean and variance vectors of the latent space.
The mean vector represents the center of the latent space, while the variance vector represents the spread or uncertainty of the data points in the latent space.
Reparameterization Trick:

The reparameterization trick is used to sample latent vectors from the learned mean and variance vectors in a differentiable manner.
It introduces randomness into the sampling process while maintaining differentiability, which enables efficient training of the VAE using gradient-based optimization algorithms.
The sampled latent vectors are the key component that allows the VAE to generate new data points during the decoding process.
Decoder Block:

The decoder block is responsible for reconstructing the input image from the sampled latent vectors.
It consists of a series of transposed convolutional layers followed by activation functions (e.g., ReLU) and a final sigmoid activation function.
These layers reverse the process of the encoder block by gradually increasing the spatial dimensions and reducing the number of channels of the latent representation.
The final output of the decoder block is the reconstructed image tensor, which ideally should closely resemble the original input image.
"""
########################################
class DecodeBlock(nn.Module):
    def __init__(self, decoder, flat_dim,inter_dim,  latent_dim, out_encod_dim):
        super(DecodeBlock, self).__init__()
        self.inter_dim = inter_dim
        self.flat_dim = flat_dim
        self.out_encod_dim=out_encod_dim

        # Define linear layers
        self.fc1 = nn.Linear(latent_dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, flat_dim) 
        
        self.decoder = decoder

    def forward(self, z):
        # double linear layers to obtain flat dimension
        x_flat = F.relu(self.fc2(self.fc1(z)))

        # Reshape to 4D tensor
        ch,hw=self.out_encod_dim[1], self.out_encod_dim[2]

        x_reshape = x_flat.view(-1, ch, hw, hw)  # reshape the latent representation for the decoder 
        # decode through decoder layers
        #x_reshape=self.decoder(x_reshape) ## uncomment one you train VAE only
        #print("decoded ",x_reshape.shape)
        return x_reshape# reconstructed_output
########################################
