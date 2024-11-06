from importlib import reload
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
import tensorflow as tf
from torch.nn.utils.parametrizations import weight_norm
import copy
import scipy.optimize 
import scipy.io
import pickle
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
import gc
gc.collect()
import psutil

import Modules.MultiHeadSpatialAttention 
reload(Modules.MultiHeadSpatialAttention)  # mandatory to reload content at each re-call atfer modification
from Modules.MultiHeadSpatialAttention import *

import Modules.encoder_block
reload(Modules.encoder_block)  # mandatory to reload content at each re-call atfer modification
from Modules.encoder_block import *

import Modules.decoder_block
reload(Modules.decoder_block)  # mandatory to reload content at each re-call atfer modification
from Modules.decoder_block import *

import Modules.VAE
reload(Modules.VAE)  # mandatory to reload content at each re-call atfer modification
from Modules.VAE import *


import Modules.loss_PDE
reload(Modules.loss_PDE)  # mandatory to reload content at each re-call atfer modification
from Modules.loss_PDE import *


os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.manual_seed(66) 
np.random.seed(66)
torch.set_default_dtype(torch.float32)



# generalized version
# def initialize_weights(module):
#     ''' starting from small initialized parameters '''
#     if isinstance(module, nn.Conv2d):
#         c = 0.1
#         module.weight.data.uniform_(-c*np.sqrt(1 / np.prod(module.weight.shape[:-1])),
#                                      c*np.sqrt(1 / np.prod(module.weight.shape[:-1])))

#     elif isinstance(module, nn.Linear):
#         module.bias.data.zero_()


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        #nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        c = 1 #0.5
        module.weight.data.uniform_(-c * np.sqrt(1 / (3 * 3 * 320)), 
                                    c * np.sqrt(1 / (3 * 3 * 320)))
    elif isinstance(module, nn.Linear):
        if module.bias is not None:
            module.bias.data.zero_()
#################################################
def xavier_initialization(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight) 
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )
        
    def forward(self, x):
        return x + self.block(x)

# ANN class with residual blocks
class ANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, lb, ub, dropout_prob=0.05):
        super(ANN, self).__init__()
        
        self.lb = torch.tensor(lb, dtype=torch.float32).squeeze()
        self.ub = torch.tensor(ub, dtype=torch.float32).squeeze()
        
        self.layers = [input_dim] + hidden_layers + [1]
        
        self.linear_layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # Add linear layers and residual blocks
        for i in range(len(self.layers) - 1):
            self.linear_layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:
                self.residual_blocks.append(ResidualBlock(self.layers[i + 1]))
                self.dropout_layers.append(nn.Dropout(p=dropout_prob))
        
        # Initialize weights using Xavier initialization
        self.apply(self.xavier_initialization)

    def xavier_initialization(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x, flag_print=False, activation='tanh'):
        # Normalize input
        H = (x - self.lb) / (self.ub - self.lb)

        for i, layer in enumerate(self.linear_layers[:-1]):
            H = layer(H)
            if i < len(self.residual_blocks):
                H = self.residual_blocks[i](H)
            H = torch.tanh(H)
            #if i < len(self.dropout_layers):
            #    H = self.dropout_layers[i](H)
        
        # Output layer
        Y = self.linear_layers[-1](H)
        
        # Apply the chosen activation function
        if activation == 'sigmoid':
            Y = torch.sigmoid(Y)
        elif activation == 'tanh':
            Y = 0.5 * (torch.tanh(Y) + 1.0)
        elif activation == 'relu':
            Y = torch.clamp(torch.relu(Y), min=0.0, max=1.0)
        elif activation == 'softplus':
            Y = torch.sigmoid(torch.nn.functional.softplus(Y))
        elif activation == 'hard_sigmoid':
            Y = torch.clamp(0.2 * Y + 0.5, min=0.0, max=1.0)

        return torch.clamp(Y, min=0.0, max=1.0)
##############################################################
###############
class VAE_convLSTM(nn.Module):
    def __init__(self, input_tensor, input_channels, hidden_channels, 
                input_kernel_size, input_stride, input_padding, inter_dim, latent_dim,
                sigma, mu, delta_g, eta, Nx,Ny,Nt,dt, dx,lb,ub, num_layers, upscale_factor,
                Nsteps, list_steps,time_batch_size):
        
        super(VAE_convLSTM, self).__init__()
        
        use_float64 = False 
    
        self.dtype =  torch.float64 if use_float64 else torch.float32
                
        self.input_tensor = input_tensor
        self.input_size=input_tensor.shape
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.inter_dim = inter_dim
        self.latent_dim = latent_dim
        self.sigma = sigma
        self.mu = mu
        self.delta_g = delta_g
        self.eta = eta
        self.Nx=Nx
        self.Ny=Ny
        self.Nt=Nt
        self.dt = dt
        self.dx = dx
        self.dy=dx
        self.lb=lb,
        self.ub=ub,
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)
        self.lb = torch.tensor(self.lb, dtype=self.dtype).squeeze()
        self.ub = torch.tensor(self.ub, dtype=self.dtype).squeeze()
        self.num_layers = num_layers
        self.upscale_factor = upscale_factor
        self.Nsteps = Nsteps
        self.list_steps = list_steps
        self.time_batch_size = time_batch_size
        dropout_prob=0.5
        self.dropout = nn.Dropout(dropout_prob)
        self.gamma =  nn.Parameter(torch.tensor(0.9))  # Learnable parameter

        # number of layers
        self.num_encoder = num_layers[0]
        self.num_decoder = num_layers[0]
        self.num_convlstm = num_layers[1]
        self.input_tensor=input_tensor.to(self.dtype)
        
        # Encoder
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
        encoder = nn.Sequential(*encoder_layers)   
        self.encoder=encoder 
        
        #self.SA = SpatialAttention(self.input_size) #SpatialAttention
        self.MHSA = MultiHeadSpatialAttention(self.input_size)
        
        # ConvLSTM ==> temporal evolution
        convLSTM_layers = nn.ModuleList()
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            convlstm_cell = ConvLSTMCell(
                input_channels=self.input_channels[-2],
                hidden_channels=self.hidden_channels[-1],
                input_kernel_size=self.input_kernel_size[-1],
                input_stride=self.input_stride[-1],
                input_padding=self.input_padding[-1]
            )
            convLSTM_layers.append(convlstm_cell)
        convLSTM = nn.Sequential(*convLSTM_layers)   
        self.convLSTM= convLSTM             
        
        self.flat_dim =  input_tensor.flatten(1).shape[1]
        self.inter_dim = inter_dim
        self.latent_dim = latent_dim
        
        #  linear layers
        self.fc1 = nn.Linear(self.flat_dim, self.inter_dim)
        self.fc_mu = nn.Linear(self.inter_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.inter_dim, self.latent_dim)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc_mu.weight)
        init.xavier_uniform_(self.fc_logvar.weight)        
        
        # instance of the EncodeBlock
        #self.Encode_LSTM = EncodeBlock_convLSTM(encoder, convLSTM, self.flat_dim, self.inter_dim, self.latent_dim)
    
        #self.Temporal_Attention= TemporalAttention(hidden_channels[-1])
        
        # Decoder
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
            
        decoder = nn.Sequential(*decoder_layers)        
        self.Decode = DecodeBlock(decoder, self.flat_dim, self.inter_dim,self.latent_dim,self.encoder(self.input_tensor).shape)
        
        #self.linear_reshape = LinearReshape(input_channels=128*8*8, output_channels=64*8*8)
        
        self.decoder=decoder
        
        self.Decode = DecodeBlock(self.decoder, self.flat_dim, self.inter_dim, self.latent_dim, self.encoder(self.input_tensor).shape)
        
        self.ANN = ANN(input_dim = 3, hidden_layers = [128, 128, 128,128,128,128],lb=self.lb,ub=self.ub)

        self.apply(xavier_initialization)  
    #################
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std   
    #################
    def calculate_vae_loss(self, reconstruction, x, mu=None, logvar=None, VAE=True,beta=1):
    
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction='mean')
        if VAE==False:
            return reconstruction_loss
            
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  

        total_loss = reconstruction_loss + beta * kl_loss
        return total_loss

    ################# 
    def compute_gradient_interface_map(self,x):
        """
        Computes an interface map based on the gradient magnitude of the input tensor.
        Args:
            x (torch.Tensor): Input tensor, typically an image or spatial field [batch_size, channels, height, width].
        Returns:
            torch.Tensor: A map highlighting regions of high gradient.
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=self.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=self.dtype, device=x.device).view(1, 1, 3, 3)
        
        gradient_x = torch.nn.functional.conv2d(x, sobel_x, padding=1)
        gradient_y = torch.nn.functional.conv2d(x, sobel_y, padding=1)
        
        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        threshold = gradient_magnitude.mean() + gradient_magnitude.std()  
        interface_map = gradient_magnitude > threshold
        interface_map = interface_map#.float()  
        return interface_map
    ######################################################
    def loss_data(self, predicted, true):
            """
            Calculate MSE for the data loss over specified interfacial regions.
            """
            # global loss
            global_loss =  mse_loss(predicted, true) 

            # Extract interfacial regions 
            #predicted_interfacial, true_interfacial = self.extract_interfacial_regions(predicted, true)
            #interfacial_loss = self.mae_loss(predicted_interfacial, true_interfacial)

            return global_loss #+ interfacial_loss
    ######################################################
    #"""
    def forward(self, initial_state, x, batch_idx=0, N_current=1, t_current=0, VAE=False):
        internal_state = []
        outputs = []
        convLSTM_previous = []
        encoder_previous = []
        outputs_latent = []
        outputs_ann = []
        second_last_state = None
        loss_vae = torch.tensor(0, dtype=self.dtype)
        #self.alpha = nn.Parameter(torch.tensor(0.9))  # Start with equal weighting
        process = psutil.Process()

        _, _, Nx, Ny = x.shape
        x_ = torch.linspace(self.lb[0], self.ub[0], Nx, dtype=self.dtype, requires_grad=True) 
        y_ = torch.linspace(self.lb[1], self.ub[1], Ny, dtype=self.dtype, requires_grad=True)
        X, Y = torch.meshgrid(x_, y_, indexing='ij')
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        x_ann = None

        for time_step in range(N_current - 1):
            t_current = (time_step + 1) * self.dt
            time_vector = torch.full_like(X_flat, t_current, dtype=self.dtype, requires_grad=True)
            g = torch.cat((X_flat.unsqueeze(1), Y_flat.unsqueeze(1), time_vector.unsqueeze(1)), dim=1)
            x_ann = self.ANN(g)[:, 0].reshape(x.shape)

            t_previous = time_step * self.dt
            time_vector_prev = torch.full_like(X_flat, t_previous, dtype=self.dtype, requires_grad=True)
            g_prev = torch.cat((X_flat.unsqueeze(1), Y_flat.unsqueeze(1), time_vector_prev.unsqueeze(1)), dim=1)
            x_ann_prev = self.ANN(g_prev)[:, 0].reshape(x.shape)

            if time_step == N_current - 2:
                outputs_ann.append(x_ann)

            xt = x

            # Step 2: Memory after encoding
            x = self.encoder(x.float())
            x_latent = x.clone().detach()
            encoder_previous.append(x)
            encoder_outputs = torch.cat(encoder_previous, dim=0)

            for i in range(self.num_convlstm):
                if time_step == 0:
                    h = initial_state[i]
                    internal_state.append(h)
                else:
                    h = internal_state[i]

                convlstm_cell = self.convLSTM[i]
                x, h_new = convlstm_cell(x, h)
                internal_state[i] = h_new

            convLSTM_previous.append(x)
            convLSTM_hidden_states = torch.cat(convLSTM_previous, dim=0)

            # Step 3: Memory after LSTM computation

            if VAE:
                x_flat = torch.flatten(x, start_dim=1)
                x_inter = F.relu(self.fc1(x_flat))
                mu = self.fc_mu(x_inter)
                logvar = self.fc_logvar(x_inter)
                z = self.reparameterize(mu, logvar)

                x_flat_without_lstm = torch.flatten(x, start_dim=1)
                x_inter_without_lstm = F.relu(self.fc1(x_flat_without_lstm))
                mu_vae = self.fc_mu(x_inter_without_lstm)
                logvar_vae = self.fc_logvar(x_inter_without_lstm)
                z_vae = self.reparameterize(mu_vae, logvar_vae)

                x = self.Decode(z)
                x_vae = self.Decode(z_vae)

            x = self.decoder(x)

            if VAE:
                reconstructions = self.decoder(x_vae)
                if time_step >= 0 and batch_idx == 0:
                    get_vae_loss = self.calculate_vae_loss(reconstructions, x, mu_vae, logvar_vae)
                    loss_vae = loss_vae + get_vae_loss
            else:
                reconstruction = self.decoder(self.encoder(x))
                loss_vae = loss_vae + F.mse_loss(reconstruction, x)

            # Weighted residual connection
            #x = (x_ann + x_ann_prev) / 2 + F.relu(x) 
            x = self.gamma*x_ann  + (1-self.gamma)*F.relu(x) *self.dt
            x = torch.clamp(x, 0, 1)

            if time_step == N_current - 2:
                second_last_state = internal_state.copy()

            if time_step <= N_current - 2:
                outputs.append(x)
                outputs_latent.append(x_latent)

        # Clear memory
        del xt, x_latent, x_ann, g, x_, y_, X, Y, X_flat, Y_flat
        gc.collect()

        return outputs, outputs_latent, second_last_state, loss_vae, outputs_ann
    ################################################################
    ################################################################
    def translate_grid(self, input_grid, Nx, Ny, dx, dy):
        """Translate each point in the input grid by small random deltas within the specified range."""
        # Define small translation deltas within the grid spacing
        max_translation_x = int((1 / dx) * 0.02)  # 10% of the grid spacing in x direction
        max_translation_y = int((1 / dy) * 0.02)  # 10% of the grid spacing in y direction
        
        translation_x = torch.randint(-max_translation_x, max_translation_x + 1, (Nx, Ny))
        translation_y = torch.randint(-max_translation_y, max_translation_y + 1, (Nx, Ny))

        x_coords = torch.arange(0, Nx).unsqueeze(1).expand(Nx, Ny)
        y_coords = torch.arange(0, Ny).unsqueeze(0).expand(Nx, Ny)

        # Compute new coordinates with translations
        new_x_coords = torch.clamp(x_coords + translation_x, 0, Nx - 1)
        new_y_coords = torch.clamp(y_coords + translation_y, 0, Ny - 1)

        # Flatten the coordinates for indexing
        flat_new_x_coords = new_x_coords.view(-1)
        flat_new_y_coords = new_y_coords.view(-1)
        flat_input_grid = input_grid.view(-1)

        # Index the flattened grid with the new coordinates
        indices = flat_new_x_coords * Ny + flat_new_y_coords
        translated_flat_grid = flat_input_grid[indices]
        translated_grid = translated_flat_grid.view(Nx, Ny)

        return translated_grid
    ################################################################
    def compute_and_normalize_gradients(self, loss, weights, l0, layer):

        gw = []
        for i in range(len(loss)):
            dl = torch.autograd.grad(weights[i] * loss[i], layer.parameters(), retain_graph=True, create_graph=True)[0]
            with torch.no_grad(): 
                gw.append(torch.norm(dl))  # Calculate L2 norm of the gradient
        gw = torch.stack(gw)  

        with torch.no_grad():
            # Compute the relative loss change (loss_ratio)
            loss_ratio = loss.detach() / l0
            
            # Compute the score values (rt) and gradient norms using self.alpha
            rt = loss_ratio / loss_ratio.mean()
            gw_avg = gw.mean().detach()  # Average gradient norm
            constant = (gw_avg * rt ** self.alpha).detach()  # The target gradient norm
        
        # Compute the GradNorm loss as the sum of the differences between gradient norms and the target
        gradnorm_loss = torch.abs(gw - constant).sum()
        
        # Update weights based on GradNorm loss
        #self.update_weights(weights, gradnorm_loss, gw, constant)
        
        return gradnorm_loss, gw, constant
    ################################################################
    def update_weights(self, weights, gw, constant):
        with torch.no_grad():
            for i in range(len(weights)):
                # Calculate the updated weight following the formula
                hat_lambda_i = weights[i] * (constant[i] / gw[i])
                
                # Apply the update rule
                weights[i] = (1 - self.beta) * weights[i] + self.beta * hat_lambda_i

    ################################################################
    def train(self, new_autoencoder,input, initial_state, n_iters, learning_rate,save_path,model_ann_save_path,\
            TL_VAE_convLSTM=False,TL_VAE=False,get_data=False,alpha=0.16,opt="ori", log=False):

        # Paths to the saved encoder and decoder checkpoints
        encoder_checkpoint_path = 'models/Encode_checkpoint.pt'
        decoder_checkpoint_path = 'models/Decode_checkpoint.pt'
        save_path_figs = 'figures'
        global clone_input
        clone_input=input
        train_loss_list = []
        second_last_state = []
        prev_output = []
        
        batch_loss = 0.0
        Threshold =3e-2 #e-3
            
        # Optimizer for the defaut model  
        lbfgs_optimizer =torch.optim.LBFGS(self.parameters(),lr=learning_rate, max_iter=50) 
        optimizer_Adam= optim.Adam(self.parameters(), lr=learning_rate)
        
        # gradnorm ANN
        self.ic = torch.tensor(1.0, requires_grad=False,)
        self.motion = torch.tensor(1.0, requires_grad=False)
        self.pde = torch.tensor(1.0, requires_grad=False)
        self.ic_lstm = torch.tensor(0.01, requires_grad=False)
        self.energ = torch.tensor(1.5)   
        self.data = torch.tensor(1.0, requires_grad=False)
        self.alpha = 0.5    # GradNorm alpha parameter
        self.beta = 0.01  # Learning rate  
        
        self.optimizer = torch.optim.Adam([
            {'params': self.ANN.parameters()},  # Model parameters
            {'params': [self.ic, self.motion]}  # Learnable weights
        ], lr=1e-4)
        
        optimizer_ann= optimizer_Adam#optim.Adam(self.ANN.parameters(), lr=learning_rate)
        optimizer_convLSTM= optim.Adam(self.convLSTM.parameters(), lr=learning_rate)
        
        self.optimizer=  optimizer_Adam
        self.ANN.optimizer=optimizer_ann #

        self.convLSTM.optimizer=optimizer_convLSTM
        
        scheduler = StepLR(self.optimizer, step_size=100, gamma=0.97)

        # init dict to store losses, weights, and coefficients
        tracking_dict = {
            'epoch': [],
            'combined_loss': [],
            'weights': {
                'ic': [],
                'motion': [],
                'pde': [],
                'data': []
            },
            'coefficients': {
                'alpha': [],
                'beta': [],
                'gamma': []
            }
        }
        
        input=torch.load('grund_th_1501.pt').to(dtype=self.dtype)[0].unsqueeze(0)
        tf.print("input",input.shape)
        
        grund_th = torch.load('grund_th_1501.pt').to(dtype=self.dtype)[::int(self.ub[2]/self.Nt)]
        tf.print("grund_th",grund_th.shape)
        
        u1_true=torch.load('grund_th_1501.pt').to(dtype=self.dtype)[0].unsqueeze(0) 
        tf.print("u1_true",u1_true.shape)            
            
        # load previous model if available 
        if TL_VAE_convLSTM==True:
            #self=torch.load(save_path)
            self, self.optimizer, scheduler = self.load_checkpoint(self, self.optimizer, scheduler, save_path)
            tf.print(" Transfer of Learning done ==> PINN")
            self.real_time_process(input,initial_state,None,0, self.time_batch_size,self.dt, grund_th) 

        if TL_VAE==True:
            encoder_optim=optim.Adam(self.encoder.parameters(), lr=0.0001)
            decoder_optim=optim.Adam(self.Decode.decoder.parameters(), lr=0.0001)
            
            self.encoder, _, _       =self.load_checkpoint(self.encoder,        encoder_optim,  scheduler, encoder_checkpoint_path)
            self.Decode.decoder, _, _=self.load_checkpoint(self.Decode.decoder, decoder_optim,  scheduler, decoder_checkpoint_path)
            tf.print("\n --- Transfer of Learning done ==> encoder & decoder --- \n")
            self.real_time_process(input,initial_state,outputs_ann,0,self.Nsteps, self.time_batch_size,self.dt)
            tf.print(" \n\n")        
        
        # Training intervals
        training_intervals = self.Nsteps // self.time_batch_size
        if self.Nsteps % self.time_batch_size != 0:
            training_intervals += 1
        tf.print("Total number of training intervals:", training_intervals)  
        retain_graph=False #if training_intervals > 1 else False 
            
        latent_channels= self.encoder(input).shape[1]
        Nx,Ny,Nt,dt, dx, mu, sigma, eta, delta_g=self.Nx,self.Ny,self.Nt, self.dt, self.dx, self.mu, self.sigma, self.eta, self.delta_g
        
        loss_PDE_instance = loss_PDE(Nx,Ny,Nt,dt, dx, mu, sigma, eta, delta_g, input.shape[1],latent_channels , opt) 
        loss_PDE_instance_L = loss_PDE(Nx,Ny,Nt,dt, dx, mu, sigma, eta, delta_g, input.shape[1],latent_channels ,"latent") 
        
        hidden_state = initial_state
        global outputs_ann_current,N_current
        outputs_ann_current = torch.zeros(self.Nsteps+1, 1,self.Nx, self.Ny)
        tf.print("outputs_ann_current",outputs_ann_current.shape)
        
        ################################################################  
        ################################################################  
        ################    Main train  loop           #################    
        ################################################################
        ################################################################
        process = psutil.Process()
        outputs_ann_current[0]=input
        N_current=2 # start with t_0, t_1 as time steps , then progressively until covers all Nsteps
        flag_down_ANN=False  # by default start training ANN
        
        while N_current < self.Nsteps:  # N_current .. the marching step of ANN 
            global incr
            incr=0 
            epoch_scipy=0
            global g_data
            g_data=True 
            global epoch 
            global flag_IC
            flag_IC=True 
            for epoch in range(n_iters):
                
                t_current=(incr+1)*self.dt

                running_loss = 0
                running_pde_loss = 0
                running_Energy_loss = 0
                running_pde_L_loss = 0
                running_IC_loss = 0
                running_vae_loss = 0
                running_data_loss = 0
                weighted_running_loss = 0 
                running_ann_loss= 0
                running_diff_loss= 0
                running_colloc_loss= 0

                for time_batch_idx in range(0, N_current, self.time_batch_size):
                    self.optimizer.zero_grad()
                    self.ANN.optimizer.zero_grad()

                    input_batch = input  

                    phi_0 = input_batch if time_batch_idx == 0 else prev_output[-2:-1].detach()

                    phi_0_latent = self.encoder(phi_0)

                    # get relevant output tensors of the model
                    known_solution=grund_th[:N_current-1]
                    # this grund truth is the current (cover the original grund_th until N_current), to not confuse with the original
                    global Grund_truth                    
                    Grund_truth = torch.load('grund_th_1501.pt').to(dtype=self.dtype)[::int(self.ub[2]/self.Nt)][:N_current].detach().clone()
                    
                    known_solution_latent = self.encoder(known_solution)         
                    output, output_latent, second_last_state, loss_vae,outputs_ann = self.forward(hidden_state, phi_0, time_batch_idx,N_current,t_current)
                    
                    output = torch.cat((phi_0, torch.cat(output, dim=0)), dim=0)
                
                    # ANN
                    if flag_down_ANN == False: 
                        outputs_ann =torch.cat((phi_0, torch.cat(outputs_ann, dim=0)), dim=0)
                        outputs_ann_current[incr+1]=outputs_ann[-1].unsqueeze(0).clone()
                        outputs_ann_copy = outputs_ann_current[:incr+2].clone().detach().requires_grad_(True)
                        loss_ann, phi2D_ann, loss_IC_ann, grid_motion_loss, data_loss, loss_E_phi =self.get_physical_Loss_original_dim_int(outputs_ann_copy,t_current,Grund_truth)        
                        outputs_ann_copy = torch.cat([outputs_ann_copy[:-1], phi2D_ann.clone().detach()], dim=0)                    
                        
                    # latent
                    output_latent = torch.cat((phi_0_latent, torch.cat(output_latent, dim=0)), dim=0)
                    output_latent = output_latent[:-1]
                    u_1_pred = output[1].unsqueeze(0).clone().detach().requires_grad_(True) 
                    loss_pde_L =loss_PDE_instance_L.compute_loss(output,output_latent,outputs_ann) if len(output) >=3 else torch.zeros_like(loss_vae)#                    
                    if epoch % 10 == 0:
                        self.save_figs_ann(outputs_ann_copy.clone().detach(),Grund_truth,filename_micro="micro_evol_ann",radius_filename = "ann_radius_evolution")
                        if flag_down_ANN == False: 
                            self.save_figs_ann(output.clone().detach(),Grund_truth,filename_micro="micro_evol_lstm",radius_filename = "lstm_radius_evolution")
                        #self.save_figs_ann(output, filename_micro="pred_evol_truth", radius_filename="pred_truth_radius")
                    loss_pde, loss_Energy, loss_diff = loss_PDE_instance.compute_loss(output, output_latent[:, 0:latent_channels, :, :], outputs_ann_copy) # #
            
                    """
                    phi_colloc=self.translate_grid(phi_0.squeeze(0).squeeze(0), self.Nx, self.Ny, self.dx, self.dy).unsqueeze(0).unsqueeze(0)
                    output_colloc, output_latent_colloc,_, _,_= self.forward(hidden_state, phi_colloc, time_batch_idx,N_current,t_current)
                    output_colloc = torch.cat((phi_colloc, torch.cat(output_colloc, dim=0)), dim=0)
                    #plt.imshow(phi_modif.squeeze(0).squeeze(0))
                    #plt.savefig("phi_colloc")
                    #plt.close()
                    if  N_current<=1:
                        loss_pde_colloc, loss_Energy_colloc, loss_diff_colloc =torch.zeros_like(loss_vae),torch.zeros_like(loss_vae),torch.zeros_like(loss_vae)
                    else:
                        loss_pde_colloc, loss_Energy_colloc, loss_diff_colloc = loss_PDE_instance.compute_loss(output_colloc, output_latent[:, 0:latent_channels, :, :],outputs_ann,outputs_ann_copy,Grund_truth) # 
                    """
                    loss_pde_colloc=torch.zeros_like(loss_vae)
                    if time_batch_idx == 0:
                        loss_IC =torch.zeros_like(loss_vae)##self.loss_IC(u_1_pred, u1_true) #                    
                    # grad norm 
                    if flag_down_ANN == False: 
                        loss = loss_ann + self.pde * (loss_pde + loss_pde_L) + loss_vae +loss_diff+ loss_IC # + self.ic_lstm* loss_IC 
                        combined_loss = torch.stack([loss_IC_ann, grid_motion_loss,loss_pde])
                        loss_labels = ["IC_ann", "Motion", "PDE", "IC_lstm"]  # for printing 
                        if epoch == 0:
                            global weights, l0, T, optimizer_gn
                            weights = torch.tensor([1.0, 1.0, 1.0], device=combined_loss.device)
                            weights = torch.nn.Parameter(weights)
                            T = weights.sum().detach()  # sum of weights
                            optimizer_gn = torch.optim.Adam([weights], lr=1e-4)
                            l0 = combined_loss.detach()  # Initial loss values

                        weighted_loss = torch.dot(weights, combined_loss)                        
                        _,gw, constant  = self.compute_and_normalize_gradients(combined_loss, weights, l0, self.ANN)
                    
                        loss.backward(retain_graph=retain_graph)
                        with torch.no_grad():
                            # Update weights based on GradNorm logic
                            new_weights = weights * (1 - self.beta) + self.beta * (constant / gw)
                            new_weights = new_weights / new_weights.sum() * T  # renormalize the weights
                            self.ic, self.motion, self.pde= new_weights[0].detach(), new_weights[1].detach(), \
                                new_weights[2].detach()
                    else :  # conv-LSTM no more assited with ANN
                        self.pde=1 
                        self.ic_lstm=1 
                        loss= self.pde * (loss_pde+loss_pde_L)  + self.ic_lstm* loss_IC 
                        loss_ann=loss_IC_ann=grid_motion_loss= torch.zeros_like(loss_vae)
                        #retain_graph=False 
                        
                    # desactivated block for gradnorm 
                    #optimizer_gn.zero_grad()
                    #gradnorm_loss.backward()
                    #optimizer_gn.step()

                    self.optimizer.step() 
                    self.ANN.optimizer.step()
                    self.convLSTM.optimizer.step()
                    
                    if epoch %  25 == 0 and flag_down_ANN== False:
                        weights_with_labels = {label: weight.item() for label, weight in zip(loss_labels, new_weights)}
                        tf.print(f"\n  Weights at epoch {epoch}: {weights_with_labels}  ")
                        tf.print(f"\n  weighting gamma : {self.gamma} \n ")
                        
                    running_loss += loss.sum().item()
                    running_pde_loss += loss_pde.item()
                    running_Energy_loss += loss_Energy.item()
                    running_pde_L_loss += loss_pde_L.item()
                    running_IC_loss += loss_IC.item()
                    running_vae_loss += loss_vae.item()
                    running_data_loss += data_loss.item()
                    running_ann_loss +=  loss_ann.item()
                    running_diff_loss += loss_diff.item()
                    running_colloc_loss +=loss_pde_colloc.item()

                    # state and output for next batch
                    prev_output = output 
                    hidden_state = []
                    for h in second_last_state:
                        hidden_state.append(h.detach())

                    if log:
                        log_weights.append(weights.detach().cpu().numpy().copy())  # Log current weights
                        log_loss.append(loss_ratio.detach().cpu().numpy().copy())  # Log normalized loss ratios
                        
                ###############  L-BFGS-B Optimizer   ##############
                # call scipy optimizer if loss > thresh

                if epoch > 0 and epoch_scipy % 20e5==0 :# and running_loss > Threshold :   
                    #"""    
                    #loss = lbfgs_optimizer.step(closure)
                    #loss_pde = closure.loss_pde
                    #loss_Energy = closure.loss_Energy
                    #loss_ann = closure.loss_ann
                    #"""
                    global clone_input_batch, clone_hidden_state, clone_time_batch_idx, clone_N_current, clone_t_currect
                    clone_input_batch, clone_hidden_state, clone_time_batch_idx, clone_N_current, \
                        clone_t_currect = input_batch, hidden_state, time_batch_idx, N_current, t_current
                    
                    tf.print("\n")
                    tf.print("!!! Scipy optimize: !!! - Epoch: ", str(epoch))
                    global scipy_iter
                    scipy_iter = 1 
                    global list_loss_scipy
                    list_loss_scipy = []
                    model=self.ANN   # to change ANN
                    params =self.get_weights(model).detach().numpy()
                    tf.print("ANN param", len(self.get_weights(self.ANN).detach().numpy()))
                    #tf.print("convLSTM param", len(self.get_weights(self.convLSTM).detach().numpy()))
                    
                    results = scipy.optimize.minimize(
                        fun=self.optimizerfunc,             # Objective function
                        x0=params,                          # Initial parameters (converted to numpy array)
                        args=(),                            # Additional arguments for the objective function
                        method='L-BFGS-B',                  # Optimization method
                        jac=True,                           # If True, fun returns (f, g), where g is the gradient
                        callback=self.optimizer_callback,   # Callback function to monitor optimization progress
                        options={
                            'disp': None,                   # Display convergence messages
                            'maxiter': 5000000,              # Maximum number of iterations
                            'gtol': 1e-20,                  # Gradient norm must be less than gtol for convergence
                            'iprint': -1                    # Print level: -1 for no output, 0 for final result, 1 for iteration info
                        }
                    )     

                    self.set_weights(results.x,model)  

                    tf.print("!!! Scipy optimization done !!!\n ")

                    print_epoch_progress(epoch,n_iters,running_pde_loss, running_Energy_loss, running_pde_L_loss, 
                        running_vae_loss, running_IC_loss,running_ann_loss,running_diff_loss, running_data_loss, running_loss,running_colloc_loss )
                    del clone_input_batch, clone_hidden_state, clone_time_batch_idx, clone_N_current, clone_t_currect

                ###################################################      
                if epoch_scipy % 110 == 0  and flag_down_ANN== False:
                    tf.print("!!! resample IC points !!!\n ")
                    flag_IC=True 
                
                if epoch == 0 or (epoch + 1) % 200 == 0:
                    print_cpu_memory_usage("Memory usage: ")
                ###################################################    
                # Print loss in each epoch
                if epoch == 0 or (epoch + 1) % 20 == 0:
                    print_epoch_progress(epoch,n_iters,running_pde_loss, running_Energy_loss, running_pde_L_loss, 
                    running_vae_loss, running_IC_loss, running_ann_loss,running_diff_loss, running_data_loss, running_loss,running_colloc_loss )
                        
                # save PINN
                if running_loss < Threshold and epoch_scipy > 10    and (epoch+1) % 27 == 0:  
                    #Threshold=running_loss  
                    if flag_down_ANN== False:
                        outputs_ann_current[incr+1]=output[-1].unsqueeze(0).clone() # ANN updated with good conv-LSTM prediction
                        tf.print("outputs_ann: ", outputs_ann.shape)  # should be always two (current t_min - t_max)
                        tf.print("output: ", output.shape)
                        tf.print("outputs_ann_current: ", outputs_ann_current[:incr+2].shape)
                    else :
                        Threshold=running_loss 

                    # increment an reset flags and counters 
                    N_current +=1
                    incr +=1
                    N_current=min(N_current,self.Nsteps+1)
                    incr=min(N_current,incr)                       
                    epoch_scipy=0 # reset 
                    flag_IC=True  # reset the resampling of IC batch 
                    
                    if len(outputs_ann_current[:incr+1] ) ==self.Nsteps+1 and not flag_down_ANN:
                        flag_down_ANN=True
                        tf.print("\n flag_down_ANN !!!!! ",flag_down_ANN,"\n")  # no more train ANN 
                        tf.print("outputs_ann_current: ", outputs_ann_current[:incr+2].shape)

                    tf.print("\n Threshold reached at epoch ", epoch, "N_current= ",N_current, "incr: ",incr,"\n")
                    tf.print("output: ", output.shape)
                    tf.print("new Threshold: ", Threshold)
                    
                    #self.release_memory_with_checkpoint( scheduler,optimizer_ann, model_ann_save_path)
                
                    # Threshold=running_loss


                if  (epoch+1) % 1000 == 0 and flag_down_ANN== False: # periodic saving 
                    outputs_ann_current[incr+1]=outputs_ann[-1].unsqueeze(0).clone()
                    self.save_checkpoint(self, self.optimizer, scheduler, save_path)
                    self.real_time_process(input,initial_state,None,0, self.time_batch_size,self.dt, grund_th) 
                    
                if epoch % 50 == 0 and flag_down_ANN==False: # periodic saving 
                    tracking_dict['epoch'].append(epoch)
                    tracking_dict['combined_loss'].append(combined_loss.detach().cpu().numpy())
                    tracking_dict['weights']['ic'].append(self.ic.item())
                    tracking_dict['weights']['motion'].append(self.motion.item())
                    tracking_dict['weights']['pde'].append(self.pde.item())
                    tracking_dict['weights']['data'].append(self.data.item())
                    tracking_dict['coefficients']['alpha'].append(self.alpha)
                    tracking_dict['coefficients']['beta'].append(self.beta)
                    tracking_dict['coefficients']['gamma'].append(self.gamma)
                    
                    with open('training_tracking_dict.pkl', 'wb') as f:
                        pickle.dump(tracking_dict, f)
                    tf.print(f"\n Epoch {epoch}: Tracking dictionary updated and saved to disk. \n")
                    
                epoch_scipy +=1
                
                # Clear memory
                del input_batch, phi_0, phi_0_latent, known_solution_latent, output, output_latent, second_last_state, outputs_ann, u_1_pred, loss_pde_L, loss_ann, loss_pde, loss_Energy, loss_diff, loss_pde_colloc, loss_IC, loss

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # get logs
            if log:
                return np.stack(log_weights), np.stack(log_loss)
    #######################################################################
    def get_weights(self,model):
        """
        Get all model parameters as a single 1D tensor.
        This is useful for optimizers that operate on flattened parameters.
        """
        # Flatten all parameters into a single 1D tensor
        parameters_1d = torch.cat([param.view(-1) for param in model.parameters()])
        return parameters_1d
    #######################################################################    
    def set_weights(self, flattened_weights,model):
        """
        Set model parameters from a single 1D tensor.
        This is useful for restoring model state from a flattened parameter vector.
        """
        flattened_weights = torch.tensor(flattened_weights, dtype=self.dtype)

        if flattened_weights.numel() != sum(p.numel() for p in model.parameters()):
            raise ValueError("Flattened weights size does not match model parameters size")
    
        index = 0
        for param in model.parameters():
            param_elements = param.numel()
            # Extract the corresponding part of the flattened tensor and reshape it to match the parameter
            new_param_values = flattened_weights[index:index + param_elements].view_as(param)
            param.data.copy_(new_param_values)  # Copy the new values into the parameter
            index += param_elements
    ##############################################################################
    def optimizer_callback(self,parameters):
        global scipy_iter
        global list_loss_scipy
        if scipy_iter % 10 == 0:
            tf.print('Iter: {0:d}, total_loss: {1:.3e}'.format(scipy_iter, list_loss_scipy[-1]))

        scipy_iter += 1

        return  list_loss_scipy  
    ###############################################
    def closure(self, input_batch, hidden_state, time_batch_idx, N_current, t_current):
        global outputs_ann_current, incr
        Nx, Ny, Nt, dt, dx, mu, sigma, eta, delta_g = self.Nx, self.Ny, self.Nt, self.dt, self.dx, self.mu, self.sigma, self.eta, self.delta_g
        latent_channels = self.encoder(input_batch).shape[1]
        global Grund_truth
        global weights, l0, T,optimizer_gn

        loss_PDE_instance = loss_PDE( Nx, Ny, Nt, dt, dx, mu, sigma, eta, delta_g, input_batch.shape[1], latent_channels, opt="ori")
        
        self.optimizer.zero_grad()
        self.ANN.optimizer.zero_grad()
        
        phi_0 = input_batch if time_batch_idx == 0 else prev_output[-2:-1].detach()
        phi_0_latent = self.encoder(phi_0)

        output, output_latent, second_last_state, loss_vae, outputs_ann = self.forward(hidden_state, phi_0, time_batch_idx, N_current, t_current)
        
        output_latent = torch.cat((phi_0_latent, torch.cat(output_latent, dim=0)), dim=0)
        
        output_latent = output_latent[:-1]
        output = torch.cat((phi_0, torch.cat(output, dim=0)), dim=0)
        u_1_pred = output[1].unsqueeze(0)

        loss_pde_L = torch.zeros_like(loss_vae)
        
        outputs_ann_current[incr + 1] = outputs_ann[-1].unsqueeze(0).clone()

        outputs_ann =torch.cat((phi_0, torch.cat(outputs_ann, dim=0)), dim=0)
        outputs_ann_current[incr+1]=outputs_ann[-1].unsqueeze(0).clone()
        outputs_ann_copy = outputs_ann_current[:incr+2].clone().detach().requires_grad_(True)
        loss_ann, phi2D_ann, loss_IC_ann, grid_motion_loss, loss_E_phi, data_loss =self.get_physical_Loss_original_dim_int(outputs_ann_copy,t_current,Grund_truth)        
        outputs_ann_copy = torch.cat([outputs_ann_copy[:-1], phi2D_ann.clone().detach()], dim=0)       
        
        #loss_pde, loss_Energy, loss_diff = loss_PDE_instance.compute_loss(output, output_latent[:, 0:latent_channels, :, :], outputs_ann, outputs_ann_copy)
        
        loss = loss_ann
        
        # grad norm  
        combined_loss = torch.stack([loss_IC_ann, grid_motion_loss])
        weighted_loss = torch.dot(weights, combined_loss)
        loss.backward(retain_graph=True)

        gradnorm_loss, gw, constant = self.compute_and_normalize_gradients(combined_loss, weights, l0, self.ANN)

        # Update weights based on GradNorm loss
        self.update_weights(weights, gw, constant)

        # Normalize weights
        weights = (weights / weights.sum() * T).detach()

        # Perform the usual optimization steps
        optimizer_gn.zero_grad()
        gradnorm_loss.backward()
        optimizer_gn.step()

        self.optimizer.step()
        self.ANN.optimizer.step()        

        updated_weighted_loss = torch.dot(weights, combined_loss)

        # Clear memory
        del phi_0, phi_0_latent, output, output_latent, second_last_state, outputs_ann, outputs_ann_copy, u_1_pred, loss_pde_L

        torch.cuda.empty_cache()

        return updated_weighted_loss
    ###############################################
    def closure_lstm(self, input_batch, hidden_state, time_batch_idx, N_current, t_current):
        global outputs_ann_current, incr
        Nx, Ny, Nt, dt, dx, mu, sigma, eta, delta_g = self.Nx, self.Ny, self.Nt, self.dt, self.dx, self.mu, self.sigma, self.eta, self.delta_g
        latent_channels = self.encoder(input_batch).shape[1]

        loss_PDE_instance = loss_PDE( Nx, Ny, Nt, dt, dx, mu, sigma, eta, delta_g, input_batch.shape[1], latent_channels, opt="ori")
        
        self.convLSTM.optimizer.zero_grad()
        phi_0 = input_batch if time_batch_idx == 0 else prev_output[-2:-1].detach()
        phi_0=self.translate_grid(phi_0.squeeze(0).squeeze(0), self.Nx, self.Ny, self.dx, self.dy).unsqueeze(0).unsqueeze(0)
        phi_0_latent = self.encoder(phi_0)

        output, output_latent, second_last_state, loss_vae, outputs_ann = self.forward(hidden_state, phi_0, time_batch_idx, N_current, t_current)
        
        output = torch.cat((phi_0, torch.cat(output, dim=0)), dim=0)
        outputs_ann = torch.cat((phi_0, torch.cat(outputs_ann, dim=0)), dim=0)
        output_latent = torch.cat((phi_0_latent, torch.cat(output_latent, dim=0)), dim=0)
        
        output_latent = output_latent[:-1]
        u_1_pred = output[1].unsqueeze(0)

        loss_pde_L = torch.zeros_like(loss_vae)
        
        outputs_ann_current[incr + 1] = outputs_ann[-1].unsqueeze(0).clone()

        self.convLSTM.train()
        #loss_ann = self.get_physical_Loss_original_dim_int(outputs_ann, t_current)
        outputs_ann_copy = outputs_ann_current[:incr + 2].clone().detach().requires_grad_(True)
        loss_pde, loss_Energy, loss_diff = loss_PDE_instance.compute_loss(output, output_latent[:, 0:latent_channels, :, :], outputs_ann, outputs_ann_copy)
        
        loss = loss_pde+ loss_Energy+ loss_diff

        loss.backward()

        # Clear memory
        del phi_0, phi_0_latent, output, output_latent, second_last_state, outputs_ann, outputs_ann_copy, u_1_pred, loss_pde_L, loss_pde, loss_Energy, loss_diff

        torch.cuda.empty_cache()

        return loss
    ###############################################    
    def optimizerfunc(self, parameters): #
        global clone_input_batch, clone_hidden_state, clone_time_batch_idx, clone_N_current, clone_t_currect
        global list_loss_scipy
        
        self.set_weights(parameters,self.ANN)
        
        self.ANN.optimizer.zero_grad()
        
        loss = self.closure(clone_input_batch, clone_hidden_state, clone_time_batch_idx, clone_N_current, clone_t_currect)

        gradients = []
        for param in self.ANN.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        
        if not gradients:
            raise RuntimeError("No gradients computed. Check if the ANN parameters are being used in the loss computation.")
        
        flattened_gradients = torch.cat(gradients).detach().numpy()

        list_loss_scipy.append(loss.item())

        return loss.item(), flattened_gradients
    ############################################### 
    ###############################################    
    def optimizerfunc_convLSTM(self, parameters): #
        global clone_input_batch, clone_hidden_state, clone_time_batch_idx, clone_N_current, clone_t_currect
        global list_loss_scipy
        
        self.set_weights(parameters,self.convLSTM)
        
        self.convLSTM.optimizer.zero_grad()
        
        loss = self.closure_lstm(clone_input_batch, clone_hidden_state, clone_time_batch_idx, clone_N_current, clone_t_currect)

        gradients = []
        for param in self.convLSTM.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        
        if not gradients:
            raise RuntimeError("No gradients computed. Check if the ANN parameters are being used in the loss computation.")
        
        flattened_gradients = torch.cat(gradients).detach().numpy()

        list_loss_scipy.append(loss.item())

        return loss.item(), flattened_gradients 
    ###############################

    ###############################
    def generate_collocation_points(self, batch_XYT, batch_phi, x_max, x_min, y_max, y_min, t_current, number_points_per_batch=10):
        t_min = t_current -self.dt
        t_max = t_current 
        R0 = self.eta / 3
        
        batch_X_f = []
        
        for index, point in enumerate(batch_XYT):
            x, y, t = point
            
            for _ in range(number_points_per_batch):
                theta = random.uniform(0, 2 * math.pi)
                radius = R0 * math.sqrt(random.uniform(0, 1))
                collocation_x = x + radius * math.cos(theta)
                collocation_y = y + radius * math.sin(theta)
                collocation_x = max(min(collocation_x, x_max), x_min)
                collocation_y = max(min(collocation_y, y_max), y_min)
                collocation_t = random.uniform(t_min, t_max)

                batch_X_f.append([collocation_x, collocation_y, collocation_t])

        batch_X_f = torch.tensor(batch_X_f, dtype=self.dtype)
        return batch_X_f
    ###############################
    def get_free_energy(self,eta, phi, dphi_dt, sigma, delta_g, dx, dy):
        
        dphi_dx = (torch.roll(phi, shifts=-1, dims=1) - torch.roll(phi, shifts=1, dims=1)) / (2 * dx)
        dphi_dy = (torch.roll(phi, shifts=-1, dims=0) - torch.roll(phi, shifts=1, dims=0)) / (2 * dy)
    
        dphi_dt_dx = (torch.roll(dphi_dt, shifts=-1, dims=1) - torch.roll(dphi_dt, shifts=1, dims=1)) / (2 * dx)
        dphi_dt_dy = (torch.roll(dphi_dt, shifts=-1, dims=0) - torch.roll(dphi_dt, shifts=1, dims=0)) / (2 * dy)
    
        dot_product = dphi_dx * dphi_dt_dx + dphi_dy * dphi_dt_dy
    
        term1 = (4 * sigma / eta) * (dphi_dt * (1 - 2 * phi))
        term2 = (4 * sigma / eta) * ((2 * eta**2 / torch.pi**2) * dot_product)
    
        F = torch.sum(term1 + term2) * dx * dy  #  over the domain
        
        return F
    ###############################
    def compute_gradients_2(self, x, y, t, x_2D=None):
        # Ensure x, y, t require gradients
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        g = torch.cat((x, y, t), dim=1)
        
        # Forward pass through the model
        phi = self.ANN(g) 
        if x_2D is not None: 
            phi2D_ann = phi.reshape(x_2D.shape)
        # Compute gradients wrt t
        phi_t = torch.autograd.grad(outputs=phi, inputs=t, grad_outputs=torch.ones_like(phi),
                                    create_graph=True, retain_graph=True)[0]

        # Compute gradients wrt x and y
        #"""
        phi_x = torch.autograd.grad(outputs=phi, inputs=x, grad_outputs=torch.ones_like(phi),
                                    create_graph=True, retain_graph=True)[0]
        phi_y = torch.autograd.grad(outputs=phi, inputs=y, grad_outputs=torch.ones_like(phi),
                                    create_graph=True, retain_graph=True)[0]
        
        # Compute second-order derivatives (laplacian)
        phi_xx = torch.autograd.grad(outputs=phi_x, inputs=x, grad_outputs=torch.ones_like(phi_x),
                                    create_graph=True, retain_graph=True)[0].detach().requires_grad_(False)
        phi_yy = torch.autograd.grad(outputs=phi_y, inputs=y, grad_outputs=torch.ones_like(phi_y),
                                    create_graph=True, retain_graph=True)[0].detach().requires_grad_(False)  
        """
        laplace_phi= (
            torch.roll(phi2D_ann, shifts=1, dims=2) +
            torch.roll(phi2D_ann, shifts=-1, dims=2) +
            torch.roll(phi2D_ann, shifts=1, dims=3) +
            torch.roll(phi2D_ann, shifts=-1, dims=3) -
            4 * phi2D_ann
        ) / (self.dx * self.dy)
        lap_phi=laplace_phi.reshape(phi.shape)

        """
        lap_phi = phi_xx + phi_yy
        #lap_phi = phi_xx
        #lap_phi.add_(phi_yy)
        
        del phi_x, phi_y, phi_xx, phi_yy

        if x_2D is not None: 
            # In this case, reshape to 2D
            
            return phi, phi_t, lap_phi, phi2D_ann
        else:
            return phi, phi_t, lap_phi
    ###############################
    def compute_gradients(self, x, y, t, x_2D):
    
        # Ensure x, y, t require gradients
        x = x.clone().detach().requires_grad_(False)
        y = y.clone().detach().requires_grad_(False)
        t = t.clone().detach().requires_grad_(False)

        g = torch.cat((x, y, t), dim=1)
        g_prev =  torch.cat((x, y, torch.zeros_like(t)), dim=1)# torch.cat((x, y, t-self.dt), dim=1)

        # Compute phi2D_ann and phi2D_ann_prev
        phi2D_ann_prev = torch.clamp(self.ANN(g_prev)[:, 0], 0, 1).reshape(x_2D.shape)
        phi2D_ann = torch.clamp(self.ANN(g)[:, 0], 0, 1).reshape(x_2D.shape)

        # Compute the difference
        phi_diff = phi2D_ann - phi2D_ann_prev

        # Compute the derivative using autograd
        #phi_t = torch.autograd.grad(phi_diff.sum(), g, create_graph=True)[0][:, 2].reshape(x_2D.shape) / self.dt
        phi_t = torch.autograd.grad(phi_diff.sum(), t, create_graph=True)[0].reshape(x_2D.shape)
        """
        laplace_phi= (
            torch.roll(phi2D_ann_prev, shifts=1, dims=2) +
            torch.roll(phi2D_ann_prev, shifts=-1, dims=2) +
            torch.roll(phi2D_ann_prev, shifts=1, dims=3) +
            torch.roll(phi2D_ann_prev, shifts=-1, dims=3) -
            4 * phi2D_ann_prev
        ) / (self.dx * self.dy) 
        """
        grad_phi_x = torch.autograd.grad(phi2D_ann.sum(), x, create_graph=True)[0].reshape(x_2D.shape)
        grad_phi_y = torch.autograd.grad(phi2D_ann.sum(), y, create_graph=True)[0].reshape(x_2D.shape)

        # Compute the second derivatives (Laplacian components)
        laplace_phi_x = torch.autograd.grad(grad_phi_x.sum(), x, create_graph=True)[0].reshape(x_2D.shape)
        laplace_phi_y = torch.autograd.grad(grad_phi_y.sum(), y, create_graph=True)[0].reshape(x_2D.shape)

        # Sum the second derivatives to get the Laplacian
        laplace_phi = laplace_phi_x + laplace_phi_y
        #"""
        return phi2D_ann, phi_t, laplace_phi 
    ########################################################
    def loss_IC(self, target, pred):
        loss_IC = mse_loss(target, pred) 
        return loss_IC
    ########################################################
    def loss_IC_ann(self, X_ini, phi_ini):
        phi_ini_pred = self.ANN(X_ini)
        phi_ini_pred=torch.clamp(phi_ini_pred,0,1)
        loss_IC = mse_loss(phi_ini, phi_ini_pred) 

        del  X_ini, phi_ini
        return loss_IC, phi_ini_pred
    ########################################################
    def compute_loss_terms(self,x_seq, y_seq, t_seq, phi_ini=None):
        if phi_ini is not None:
            phi, phi_t, lap_phi, phi_2D = self.compute_gradients_2(x_seq, y_seq, t_seq, phi_ini.unsqueeze(0).unsqueeze(0))
        else:
            phi, phi_t, lap_phi = self.compute_gradients_2(x_seq, y_seq, t_seq)
        
        phi_term = (torch.pi**2 / self.eta**2) * (phi - 1/2)
        right_side_eqn = self.mu * self.sigma * (lap_phi + phi_term)
        f_phi = phi_t - right_side_eqn
        
        return (f_phi,phi_2D) if phi_ini is not None else f_phi 
    ########################################################
    def get_physical_Loss_original_dim_int(self, Phi, t_current, Grund_truth):
        process = psutil.Process()
        # Create grid
        lb, ub = self.lb, self.ub
        x = torch.linspace(lb[0], ub[0], self.Nx)
        y = torch.linspace(lb[1], ub[1], self.Ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        X_flat, Y_flat = X.flatten().float(), Y.flatten().float()

        phi_ini_ = Phi[Phi.shape[0] - 2, 0] 
        # Initial conditions (IC)
        global flag_IC, X_ini, phi_ini
        if flag_IC: 
            phi_ini = Phi[Phi.shape[0] - 2, 0].flatten().detach().to(self.dtype).unsqueeze(1)
            time_ini = torch.full_like(X_flat, t_current - self.dt)
            X_ini = torch.stack((X_flat, Y_flat, time_ini), dim=1)
            flag_IC = False

        # Loss calculation for IC
        loss_IC_ann, phi_ini_pred = self.loss_IC_ann(X_ini, phi_ini)

        # Flatten and detach for the current timestep
        phi_flattened = Phi[Phi.shape[0] - 2, 0].flatten().detach().to(self.dtype)

        # Grid motion loss
        time_vector = torch.full_like(X_flat, t_current)
        X_Y_T = torch.stack((X_flat, Y_flat, time_vector), dim=1)
        random_times = torch.tensor([random.uniform(t_current - self.dt, t_current) for _ in range(X_Y_T.shape[0])])
        X_Y_T[:, 2] = random_times
        
        x_seq, y_seq, t_seq = X_Y_T[:, 0:1].requires_grad_(True), X_Y_T[:, 1:2].requires_grad_(True), X_Y_T[:, 2:3].requires_grad_(True)
        #memory_info = process.memory_info()
        #tf.print(f"Epoch {epoch} - before compute_loss_terms: RSS = {memory_info.rss / 1e6} MB, VMS = {memory_info.vms / 1e6} MB") 
        # Loss terms for the physical model
        if len(Phi)>=2:
            f_phi, phi2D_ann = self.compute_loss_terms(x_seq, y_seq, t_seq, phi_ini_)
            grid_motion_loss = mse_loss(f_phi, torch.zeros_like(f_phi))
            loss_whole_seq = torch.zeros_like(grid_motion_loss)

        # Whole sequence loss (PDE terms)
        #loss_whole_seq = torch.zeros_like(grid_motion_loss)
        if len(Phi) > 2:
            phi_t = (Phi[1:] - Phi[:-1]) / self.dt
            laplace_phi = (torch.roll(Phi, 1, 2) + torch.roll(Phi, -1, 2) + torch.roll(Phi, 1, 3) + torch.roll(Phi, -1, 3) - 4 * Phi) / (self.dx * self.dy)
            right_side_eqn = self.mu * (self.sigma * (laplace_phi + (torch.pi**2 / (2 * self.eta**2)) * (2 * Phi - 1)))[:-1]
            f_phi_seq = phi_t - right_side_eqn
            loss_whole_seq=mse_loss(f_phi_seq, torch.zeros_like(f_phi_seq))
            g = torch.cat((x_seq, y_seq, t_seq), dim=1)
            phi2D_ann = self.ANN(g)[:, 0].reshape(Phi[Phi.shape[0] - 2, 0].shape).unsqueeze(0).unsqueeze(0)
            
        #memory_info = process.memory_info()
        #tf.print(f"Epoch {epoch} - after compute_loss_terms: RSS = {memory_info.rss / 1e6} MB, VMS = {memory_info.vms / 1e6} MB") 
                
        # Energy loss (PDE terms)
        loss_E_phi = torch.zeros_like(loss_whole_seq)
        if len(Phi) >= 3:
            f_E_phi, F, dF_dt, RHS, out_t = loss_PDE.get_energy_terms(self, Phi)
            loss_E_phi = mae_loss(f_E_phi, torch.zeros_like(f_E_phi))

        # Data loss (if ground truth is available)
        data_loss = mse_loss(phi2D_ann, Grund_truth[-1].unsqueeze(0)) if len(Phi) >= 0 else torch.zeros_like(grid_motion_loss)

        # Total loss calculation #grid_motion_loss
        total_loss = self.ic * loss_IC_ann + self.motion * (grid_motion_loss) + self.data * data_loss + self.energ * loss_E_phi
        
        # L1 Regularization
        l1_lambda = 1e-6
        l1_norm = sum(p.abs().sum() for p in self.ANN.parameters())
        # L2 Regularization
        l2_lambda = 1e-6
        l2_norm = sum(p.pow(2).sum() for p in self.ANN.parameters())

        total_loss = total_loss + l1_lambda * l1_norm + l2_lambda * l2_norm

        # Clean up unnecessary tensors to prevent memory leaks
        del X_Y_T, phi_flattened, phi_ini_pred,  x_seq, y_seq, t_seq
        gc.collect()


        return total_loss, phi2D_ann, loss_IC_ann, (grid_motion_loss+ loss_whole_seq), data_loss, loss_E_phi

    #######################################################################
    def save_checkpoint(self, model,optimizer, scheduler, save_dir):
        directory = os.path.dirname(save_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        '''save model and optimizer'''

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, save_dir)
    #######################################################################
    def load_checkpoint(self, model, optimizer, scheduler, save_dir):
        '''load model and optimizer'''
        #print(save_dir)
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['model_state_dict'])

        if (not optimizer is None):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        tf.print('Pretrained model loaded!')
        #print(save_dir)
        return model, optimizer, scheduler
    #######################################################################
    ##########################################################   
    def real_time_process(self,input,initial_state,outputs_ann, epoch, time_batch_size,dt,Grund_truth):
        # Initialize lists to store outputs
        all_outputs = []
        all_outputs_latent = [] 

        # Process the sequence in time batches
        for time_batch_idx in range(1): #range(0, Nsteps, time_batch_size):
            #tf.print("time_batch_idx",time_batch_idx)

            if time_batch_idx == 0:
                hidden_state = initial_state

                phi_in = input
                phi_0_latent = self.encoder(phi_in)
            else:
                hidden_state = state_detached
                print(prev_output[-1].shape)
                phi_in = prev_output[-1].detach()
                phi_0_latent = self.encoder(phi_in)
            
            output,output_latent, second_last_state,loss_vae,_ = self(hidden_state, phi_in, 0,self.Nsteps+1)
            
            output = torch.cat((input, torch.cat(output, dim=0)), dim=0)

            output_latent = torch.cat([item.unsqueeze(0) for item in output_latent], dim=0)

            all_outputs.append(output)
            all_outputs_latent.append(output_latent[:, :1])  # Keep only the first element of the time dimension
            prev_output = output
            state_detached = []
            for i in range(len(second_last_state)):
                h = second_last_state[i]
                state_detached.append(h.detach())  # hidden state
            phi_in = output[-1].unsqueeze(0).detach()  # Use the last output as the input for the next batch

        all_outputs = torch.cat(all_outputs, dim=0)  # Conv-LSTM predictions

        self.save_figs_ann(all_outputs.clone().detach(),Grund_truth,filename_micro="check_TL",radius_filename = "radius_check_TL") #  check the transfer of learning

        tf.print("real-time process, outputs shapes: ", all_outputs.shape)
        
    def save_figs(self,all_outputs,filename_micro = f"test_solution",radius_filename = f"radius_evolution"):       
        ###########################          
        dt=self.dt
        N_frames = 6
        interval_size = len(all_outputs) // N_frames
        
        fig, axes = plt.subplots(N_frames // 2, 2, figsize=(12, 12))
        
        axes = axes.flatten() if N_frames // 2 > 1 else [axes]
        
        for i, ax in enumerate(axes):
            idx = i * interval_size
            if idx < len(all_outputs)-1 :  # Include the last element check
                # Extract the image at the given index
                image = all_outputs[idx, 0].cpu().detach().numpy()  # Extract 2D image
                
                phase = ax.imshow(image, cmap='viridis')  # Plot the image
                ax.set_title(f"Sequence: {idx} - Time: {idx * dt:.2e}")  # Set title with real time
                ax.axis('off')  # Turn off axis
                fig.colorbar(phase, ax=ax)
        
        last_idx = len(all_outputs)-1
        ax = axes[-1]
        image = all_outputs[last_idx, 0].cpu().detach().numpy()
        phase = ax.imshow(image, cmap='viridis')
        ax.set_title(f"Sequence: {last_idx} - Time: {last_idx * dt:.2e}")
        ax.axis('off')

        figures_directory = 'figures'
        if not os.path.exists(figures_directory):
            os.makedirs(figures_directory)
        plt.tight_layout()
        
        filepath = os.path.join(figures_directory, filename_micro)
        plt.savefig(filepath)
        plt.close()
        ###################################  
        ###################################
        radius_vs_time = []
        for t in range(all_outputs.size(0)):
            area = torch.sum(all_outputs[t,0] > 1e-2).item()
            radius = np.sqrt(area / (self.Nx * self.Ny) / np.pi)
            radius_vs_time.append(radius)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(radius_vs_time)), radius_vs_time, marker='o', linestyle='-')
        plt.xlabel('Time Steps')
        plt.ylabel('Radius')
        plt.title('Evolution of the Radius vs Time')
        plt.grid(True)
        plt.tight_layout()
        
        radius_filepath = os.path.join(figures_directory, radius_filename)
        plt.savefig(radius_filepath)
        plt.close()
    ####################################################
    ####################################################
    def release_memory_with_checkpoint(self,scheduler, optimizer_ann,save_path_ann):
        self.save_checkpoint(self.ANN, self.ANN.optimizer, scheduler, save_path_ann)
        
        del self.ANN
        #del scheduler
        
        torch.cuda.empty_cache()  # This ensures cached memory is freed
        
        self.ANN = ANN(input_dim = 3, hidden_layers = [128, 128, 128,128,128,128],lb=self.lb,ub=self.ub) 
        self.ANN.apply(xavier_initialization)  # Reinitialize weights
        
        self.ANN,self.ANN.optimizer, scheduler = self.load_checkpoint(self.ANN, optimizer_ann, scheduler, save_path_ann)

        tf.print("Memory released for self.ANN.")
    ####################################################
    def save_figs_ann(self, all_outputs, Grund_truth, filename_micro="micro_evol_ann", radius_filename="ann_radius_evolution"):  
        import gc
        grund_truth_np =torch.load('grund_th_1501.pt').to(dtype=self.dtype)[::int(self.ub[2]/self.Nt)].detach().clone().numpy()
        ground_truth_radius_vs_time = []
        for t in range(grund_truth_np.shape[0]):
            ground_truth_area = np.sum(grund_truth_np[t, 0] > 5e-2)
            ground_truth_radius = np.sqrt(ground_truth_area / (self.Nx * self.Ny) / np.pi)
            ground_truth_radius_vs_time.append(ground_truth_radius)
                    
        radius_vs_time = []
        all_outputs_np = all_outputs.numpy()
        for t in range(all_outputs_np.shape[0]):
            area = np.sum(all_outputs_np[t, 0] > 5e-2)
            radius = np.sqrt(area / (self.Nx * self.Ny) / np.pi)
            radius_vs_time.append(radius)

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(radius_vs_time)) * int(self.ub[2]/self.Nt), radius_vs_time, marker='o', linestyle='-', label='Predicted Radius')
        plt.plot(np.arange(len(ground_truth_radius_vs_time)) * int(self.ub[2]/self.Nt), ground_truth_radius_vs_time, marker='x', linestyle='--', color='red', label='Ground Truth Radius')
        plt.xlabel('Time Steps')
        plt.ylabel('Radius')
        plt.title('Evolution of the Radius vs Time')
        plt.grid(True)
        plt.legend()  # 
        plt.tight_layout()
        radius_filename = os.path.join('figures', radius_filename)
        plt.savefig(radius_filename)
        plt.clf()
        plt.close()

        num_plots = 10  # Number of plots to generate
        fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 3, 3))

        # Calculate equally distant indices
        indices = np.linspace(1, len(radius_vs_time) - 1, num=num_plots, dtype=int)
        for i, t in enumerate(indices):
            ax = axes[i] if num_plots > 1 else axes
            ax.imshow(all_outputs[t, 0].detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
            ax.set_title(f'Time Step {t * self.dt}')
            ax.axis('off')

        plt.tight_layout()
        filename_micro = os.path.join('figures', filename_micro)
        plt.savefig(filename_micro) 
        plt.clf()
        plt.close()
        
        # Clear variables to release memory
        del radius_vs_time, all_outputs, fig, axes
        gc.collect()
################################################################
def set_initial_state(latent_representation, num_layers):
    hidden_size=64
    num_convlstm = num_layers[1]
    initial_state = []
    for _ in range(num_convlstm):

        #h0 = torch.zeros(1, latent_representation.shape[1], latent_representation.shape[2], latent_representation.shape[3])
        #h0 = torch.zeros(1, latent_representation.shape[1], latent_representation.shape[2], latent_representation.shape[3])
        h0 = torch.randn(1, hidden_size, latent_representation.shape[2], latent_representation.shape[3])
        c0 = torch.randn(1, hidden_size, latent_representation.shape[2], latent_representation.shape[3])

        init.xavier_uniform_(h0)
        initial_state.append(h0)
    return initial_state
####################################################
def train_Autoencoder(autoencoder,dataloader,save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)  

    # Initialize an empty list to store the loss values
    losses = []
    Threshold= 4.5e-2
    num_epochs = 1000

    for epoch in range(num_epochs):
        epoch_loss = 0.0 
        
        for data in dataloader:
            for sample in range(len(data)):  # Iterate over each sample in the batch
                optimizer.zero_grad()

                reconstructions = autoencoder(data[sample])
                latent_representation = autoencoder.encoder(data[sample])
                latent_shape = latent_representation.shape  # Get the shape of the latent representation
                
                loss = criterion(reconstructions, data[sample])
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()  # Accumulate the loss for the epoch

        epoch_loss /= len(dataloader.dataset)
        losses.append(epoch_loss)  # Append the epoch loss to the list
            
        if epoch % 20 == 0:
            tf.print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.3e}")
            
        # save model
        if loss < Threshold and (epoch+1) % 100 == 0:
            tf.print("Threshold reached")  
            save_checkpoint(autoencoder.encoder, optimizer,scheduler, save_dir=os.path.join("models",'encoder_checkpoint.pt'))
            save_checkpoint(autoencoder.decoder, optimizer, scheduler, save_dir=os.path.join("models",'decoder_checkpoint.pt'))
            
    # Plot the loss over epochs
    plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()  
    
    plt.savefig(os.path.join(save_path, "autoencoder_loss"))
    #print("Autoencoder trained")
    return latent_representation


################################################################

####################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_IC(PINN,input,save_path):
    with torch.no_grad():
        # Pass the single sample through the autoencoder
        tf.print("original dimension: ", input.shape)
        latent_representation = PINN.encoder(input)
        tf.print("latent dimension: ", latent_representation.shape)

        # Reconstruct the sample from the latent representation
        reconstructed_sample = PINN.decoder(latent_representation)
        tf.print("reconstructed dimension: ", reconstructed_sample.shape)

        # Convert the tensors to numpy arrays
        original_sample_np = input.numpy()
        reconstructed_sample_np = reconstructed_sample.numpy()

        # Plot original and reconstructed samples
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 6))

        # Assuming grayscale images, adjust indexing for channels
        axes[0].imshow(original_sample_np[0, 0], cmap='viridis')
        axes[0].set_title('Original')

        axes[1].imshow(latent_representation[0, 0], cmap='viridis')
        axes[1].set_title('Latent')
        
        axes[2].imshow(reconstructed_sample_np[0, 0], cmap='viridis')
        axes[2].set_title('Reconstructed')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "test_Reconstruction"))
        plt.close()    
    return latent_representation
#######################################################################
def check_IC_2(PINN, input, save_path, num_features=10):
    with torch.no_grad():
        # Pass the single sample through the autoencoder
        print("original dimension: ", input.shape)
        latent_representation = PINN.encoder(input)
        print("latent dimension: ", latent_representation.shape)

        # Reconstruct the sample from the latent representation
        reconstructed_sample = PINN.decoder(latent_representation)
        print("reconstructed dimension: ", reconstructed_sample.shape)

        # Convert the tensors to numpy arrays
        original_sample_np = input.numpy()
        reconstructed_sample_np = reconstructed_sample.numpy()

        # Plot original and reconstructed samples
        fig, axes = plt.subplots(nrows=1, ncols=num_features+2, figsize=(2*num_features, 2))

        # Plot original image
        axes[0].imshow(original_sample_np[0, 0], cmap='viridis')
        axes[0].set_title('Original')

        # Plot latent features
        for i in range(num_features):
            axes[i+1].imshow(latent_representation[0, i], cmap='viridis')
            axes[i+1].set_title(f'Latent {i+1}')

        # Plot reconstructed image
        axes[num_features+1].imshow(reconstructed_sample_np[0, 0], cmap='viridis')
        axes[num_features+1].set_title('Reconstructed')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "test_AE_reconstruct"))
        plt.close() 

    return latent_representation
#######################################################################
##############################################################################
def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)

def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))

def mse_loss(predicted, target):
    return F.mse_loss(predicted, target)

def mae_loss(predicted, target):
    return torch.mean(torch.abs(predicted - target))

def save_checkpoint_big_model(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)

##############################################################################
def print_epoch_progress(epoch, n_iters, running_pde_loss, running_Energy_loss, running_pde_L_loss, running_vae_loss, running_IC_loss, 
                        running_ann_loss,running_diff_loss, running_data_loss, running_loss,running_colloc_loss):
    tf.print(f'Epoch {epoch+1}/{n_iters} ({(epoch+1) / n_iters * 100:.2f}%) - '
            f'PDE Loss: {running_pde_loss:.3e}, '
            f'Latent Loss: {running_pde_L_loss:.3e}, '
            f'IC Loss: {running_IC_loss:.3e}, '
            f'VAE Loss: {running_vae_loss:.3e}, '
            f'ANN Loss: {running_ann_loss:.3e}, '
            f'Diff Loss: {running_diff_loss:.3e}, '
            f'Data Loss: {running_data_loss:.3e}, '
            f'Energy Loss: {running_Energy_loss:.3e}, '
            f'Colloc Loss: {running_colloc_loss:.3e}, '
            f'Original Total Loss: {running_loss:.3e}, '
            )
    tf.print("\n")
##############################################################################
def generate_outputs_from_first_sequence(model, initial_state, first_sequence_input, N_current):
    outputs = []
    outputs.append(first_sequence_input)
    _,_, Nx, Ny = first_sequence_input.shape

    x = first_sequence_input 
    for time_step in range(N_current):
        t_current = time_step * model.dt
        X_flat = torch.linspace(model.lb[0], model.ub[0], Nx, dtype=self.dtype).repeat(Ny)
        Y_flat = torch.linspace(model.lb[1], model.ub[1], Ny, dtype=self.dtype).repeat(Nx)
        time_vector = torch.full_like(X_flat, t_current, dtype=self.dtype)
        g = torch.stack((X_flat, Y_flat, time_vector), dim=1)

        with torch.no_grad():
            x_ann = model.ANN(g)
            x_ann = x_ann[:, 0].reshape(x.shape[2:])  # Reshape to (Nx, Ny)
            x_ann = x_ann.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        outputs.append(x_ann)
        first_sequence_input = x_ann.squeeze(0).squeeze(0)  # Update input for next step

    return outputs


def print_cpu_memory_usage(step):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    tf.print("!*****************************************************\n ")
    tf.print(f"Step: {step}, RSS: {memory_info.rss / 1024 ** 2} MB, VMS: {memory_info.vms / 1024 ** 2} MB\n")
    

