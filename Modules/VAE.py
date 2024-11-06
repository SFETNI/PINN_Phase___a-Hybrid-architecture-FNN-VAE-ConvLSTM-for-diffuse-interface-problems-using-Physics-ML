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


import Modules.DecodeBlock
reload(Modules.DecodeBlock)  # mandatory to reload content at each re-call atfer modification
from Modules.DecodeBlock import *


import Modules.ConvLSTMCell
reload(Modules.ConvLSTMCell)  # mandatory to reload content at each re-call atfer modification
from Modules.ConvLSTMCell import *



def initialize_weights(module):
    
    if isinstance(module, nn.Conv2d):
        #nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        c = 1 #0.5
        module.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), 
            c*np.sqrt(1 / (3 * 3 * 320)))
    
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()
########################################
class VariationaAutoencoder(nn.Module):
    def __init__(self, input_tensor, input_channels, hidden_channels, 
                input_kernel_size, input_stride, input_padding, num_layers,inter_dim,latent_dim):
        super(VariationaAutoencoder, self).__init__()
        
        self.input_channels =[input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_layers = num_layers
        self.num_encoder=num_layers[0]
        self.num_decoder=num_layers[0]
        self.input_size=input_tensor.shape
        self.flat_dim =  input_tensor.flatten(1).shape[1]
        self.inter_dim = inter_dim
        self.latent_dim = latent_dim
        
        #  linear layers
        #"""
        self.fc1 = nn.Linear(self.flat_dim, self.inter_dim)
        self.fc_mu = nn.Linear(self.inter_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.inter_dim, self.latent_dim)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc_mu.weight)
        init.xavier_uniform_(self.fc_logvar.weight)           
        #"""        
        
        
        
        encoder_layers = []
        # Loop over the encoder blocks and add them to the list
        for i in range(self.num_encoder):
            encoder_layers.append(encoder_block(
                input_channels=self.input_channels[i], 
                hidden_channels=self.hidden_channels[i], 
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i]
            ))
        self.encoder = nn.Sequential(*encoder_layers)      
        self.MHSA = MultiHeadSpatialAttention(self.input_size)        
                
        decoder_layers = []
        # Loop over the encoder blocks and add them to the list
        for i in range(self.num_decoder - 1, -1, -1):
            decoder_layers.append(decoder_block(
                input_channels=self.input_channels[i], 
                hidden_channels= self.hidden_channels[i], 
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i]
            ))
        self.decoder= nn.Sequential(*decoder_layers)    
            
        self.Decode = DecodeBlock(self.decoder, self.flat_dim, self.inter_dim,\
            self.latent_dim, self.encoder(input_tensor).shape)
                    
            
    def compute_gradient_interface_map(self,x):
        """
        Computes an interface map based on the gradient magnitude of the input tensor.
        Args:
            x (torch.Tensor): Input tensor, typically an image or spatial field [batch_size, channels, height, width].
        Returns:
            torch.Tensor: A map highlighting regions of high gradient.
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        gradient_x = torch.nn.functional.conv2d(x, sobel_x, padding=1)
        gradient_y = torch.nn.functional.conv2d(x, sobel_y, padding=1)
        
        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        threshold = gradient_magnitude.mean() + gradient_magnitude.std()  
        interface_map = gradient_magnitude > threshold
        interface_map = interface_map.float()  
        return interface_map
        
    #################
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std   

    def forward(self, x,VAE=True):
        #tf.print(" X in Auto-encoder ", x.shape)
        #interface_map = self.compute_gradient_interface_map(x)
        #x=self.MHSA(x,interface_map)
        x = self.encoder(x)
        #print("X out encoder ", x.shape)
        
        if VAE:
            x_flat = torch.flatten(x, start_dim=1)
            x_inter = F.relu(self.fc1(x_flat))
            mu = self.fc_mu(x_inter)
            logvar = self.fc_logvar(x_inter)
            z=self.reparameterize(mu, logvar)  
            x = self.Decode(z)
            
        else:
            self.fc_mu=None
            self.fc_logvar=None
            self.fc1=None 
            
        x = self.decoder(x)
            
        #tf.print("X out Auto-encoder ", x.shape)
        return x

    ###################################################
    def calculate_vae_loss(self, reconstruction, x, mu=None, logvar=None, VAE=True,beta=1):

        reconstruction_loss = F.mse_loss(reconstruction, x, reduction='mean')
        if VAE==False:
            return reconstruction_loss
            
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  

        total_loss = reconstruction_loss + beta * kl_loss
        return total_loss
    ###################################################
    def train(self,dataloader,save_path,datasets,consider_VAE=True):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.97)  


        # Initialize an empty list to store the loss values
        losses = []
        Threshold= 1e-2
        num_epochs = 10000

        for epoch in range(num_epochs):
            epoch_loss = 0.0 
            
            for data in dataloader:
                for sample in range(len(data)):  
                    optimizer.zero_grad()
                    x=data[sample]
                    reconstructions = self(x)
                    latent_representation = self.encoder(x)

                    if consider_VAE:
                        x_flat = torch.flatten(x, start_dim=1)
                        x_inter = F.relu(self.fc1(x_flat))
                        mu = self.fc_mu(x_inter)
                        logvar = self.fc_logvar(x_inter)
                        loss = self.calculate_vae_loss(reconstructions, x, mu, logvar)
                    else: 
                        loss = self.calculate_vae_loss(reconstructions, x )
                        

                    latent_shape = latent_representation.shape  

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()  

            epoch_loss /= len(dataloader.dataset)
            losses.append(epoch_loss)  
                
            if epoch % 20 == 0:
                tf.print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.3e}")

            if epoch % 20 == 0:    # periodic check        
                self.check_IC(datasets,save_path)
                
            # save model
            if loss < Threshold and (epoch+1) % 20 == 0:
                tf.print("Threshold reached")  
                Threshold=loss
                self.save_checkpoint( optimizer,scheduler, save_dir=os.path.join("models",'VAE_checkpoint.pt'))

        plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()  
        
        plt.savefig(os.path.join(save_path, "VAE_loss"))
        plt.close()
        tf.print("VAE trained")
        

    #######################################################################
    def save_checkpoint(self,optimizer, scheduler, save_dir):
        '''save model and optimizer'''

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, save_dir)

    #######################################################################
    def load_checkpoint(self, optimizer, scheduler, save_dir):
        '''load model and optimizer'''
        #print(save_dir)
        checkpoint = torch.load(save_dir)
        self.load_state_dict(checkpoint['model_state_dict'])

        if (not optimizer is None):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        tf.print('Pretrained VAE model loaded!')


        return  self, optimizer, scheduler

    #######################################################################
    def check_IC(self, datasets, save_path, num_samples=6, num_features=5):
        with torch.no_grad():
            random_indices = torch.randperm(len(datasets))[:num_samples]

            fig, axes = plt.subplots(nrows=num_samples, ncols=num_features+2, figsize=(2*num_features, 2*num_samples))

            for idx, random_index in enumerate(random_indices):
                random_sample = datasets[random_index]

                input = random_sample
                #tf.print("original dimension: ", input.shape)

                out_encoder = self.encoder(input)
                #tf.print("latent dimension: ", out_encoder.shape)
                reconstructed_sample = self(input)
                #tf.print("reconstructed dimension: ", reconstructed_sample.shape)

                original_sample_np = input.numpy()
                reconstructed_sample_np = reconstructed_sample.numpy()

                axes[idx, 0].imshow(original_sample_np[0, 0], cmap='viridis')
                axes[idx, 0].set_title('Original')

                for i in range(num_features):
                    axes[idx, i+1].imshow(out_encoder[0, i].cpu().numpy(), cmap='viridis')
                    #tf.print(out_encoder[0, i].min(), out_encoder[0, i].max())
                    axes[idx, i+1].set_title(f'Latent {i+1}')

                axes[idx, num_features+1].imshow(reconstructed_sample_np[0, 0], cmap='viridis')
                axes[idx, num_features+1].set_title('Reconstructed')

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "check_VAE"))
            plt.close()