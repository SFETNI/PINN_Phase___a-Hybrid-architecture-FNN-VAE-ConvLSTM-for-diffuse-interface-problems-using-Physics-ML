import gc
gc.collect()
import torch
#import cv2
import threading
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (non-interactive)
import tensorflow as tf
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
import scipy.ndimage
np.random.seed(1234)
tf.random.set_seed(1234)
import random#
import math 
from scipy.interpolate import NearestNDInterpolator
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)
import torch.nn.functional as F
import Modules.Conv1dDerivative
reload(Modules.Conv1dDerivative)  # mandatory to reload content at each re-call atfer modification
from Modules.Conv1dDerivative import *

import Modules.Conv2dDerivative
reload(Modules.Conv2dDerivative)  # mandatory to reload content at each re-call atfer modification
from Modules.Conv2dDerivative import *

################################################################
class loss_PDE(nn.Module):
    ''' Loss generator for physical loss '''
    ###############################
    def __init__(self,Nx,Ny,Nt, dt, dx, mu, sigma, eta, delta_g, output_ch, output_ch_latent, opt):
        ''' Construct the derivatives, X = Width, Y = Height '''
        super(loss_PDE, self).__init__()

        if opt=="latent":  # the loss is computed on the latent dimension 
            num_ch=output_ch_latent
        else:
            num_ch=output_ch
        #tf.print("output_ch,output_ch_latent", output_ch,output_ch_latent)

        # Store the parameters as attributes of the instance
        self.dt = dt
        self.mu = mu
        self.sigma = sigma
        self.eta = eta
        self.delta_g = delta_g
        self.opt=opt 
        self.Nx=Nx
        self.Ny=Ny
        self.Nt=Nt
        
        
        if opt=="latent":  # the loss is computed on the latent dimension 
            self.dx=(63/7) * dx # dx_latent increases  
            self.dy=(63/7) * dx # dx_latent increases  
        else:
            self.dx=dx
            self.dy=dx
            
        use_float64 = False 
        self.dtype = torch.float64 if use_float64 else torch.float32
        
        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = laplace_op,
            resol = (self.dx**2), 
            in_channels=num_ch,
            out_channels=num_ch,
            kernel_size = 5,
            name = 'laplace_operator')

        self.laplace_dim = Conv2dDerivative(
            DerFilter = laplace_op,
            resol = (dx**2), 
            in_channels=output_ch,
            out_channels=output_ch,
            kernel_size = 5,
            name = 'laplace_operator')

        self.dx_op = Conv2dDerivative(
            DerFilter = partial_x_sobel_op,
            resol = (self.dx*1),
            in_channels=num_ch,
            out_channels=num_ch,
            kernel_size = 7,
            name = 'dx_operator')
        
        self.dy_op = Conv2dDerivative(
            DerFilter = partial_y_sobel_op,
            resol = (self.dx*1),
            in_channels=num_ch,
            out_channels=num_ch,
            kernel_size = 7,
            name = 'dy_operator')
        
        self.dx_op_dim = Conv2dDerivative(
            DerFilter = partial_x_sobel_op,
            resol = (self.dx*1),
            in_channels=output_ch,
            out_channels=output_ch,
            kernel_size = 7,
            name = 'dx_operator')
        
        self.dy_op_dim = Conv2dDerivative(
            DerFilter = partial_y_sobel_op,
            resol = (self.dx*1),
            in_channels=output_ch,
            out_channels=output_ch,
            kernel_size = 7,
            name = 'dy_operator')      
            
        self.dt_op_Lat = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (dt*2),
            in_channels=output_ch_latent, #,
            out_channels=output_ch_latent, #
            kernel_size = 3,
            name = 'partial_t')            
            

        # temporal derivative operator
        self.dt_op = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (dt*2),
            in_channels=num_ch, #,
            out_channels=num_ch, #
            kernel_size = 3,
            name = 'partial_t')
        
        self.dt_op_dim = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (dt*2),
            in_channels=output_ch, #,
            out_channels=output_ch, #
            kernel_size = 3,
            name = 'partial_t')        

    # Methods    
    def h_term(self, phi):
        try:
            square_term = phi * (1 - phi)
            #print(square_term.min(),square_term.max())
            if torch.any(square_term < 0):
                raise ValueError("Square root term has negative values")
            square_root_term = torch.sqrt(square_term)
        except ValueError:
            raise ValueError("Cannot calculate the square root of a negative number")
        return np.pi / self.eta  * square_root_term

    ###############################
    def prepar_fft(self):
        #Nx, dx, Ny, dy = self.Nx, self.dx, self.Ny, self.dy
        Nx, dx, Ny, dy = int(self.Nx), self.dx, int(self.Ny), self.dy

        # Define k_x and k_y using torch.fft.fftfreq
        k_x = 2 * torch.tensor(torch.fft.fftfreq(Nx, dx), dtype=self.dtype)
        k_y = 2 * torch.tensor(torch.fft.fftfreq(Ny, dy), dtype=self.dtype)

        # Create meshgrid using torch.meshgrid
        k_x, k_y = torch.meshgrid(k_x, k_y)

        # Calculate k_2
        k_2 = k_x**2 + k_y**2

        return k_2
    ###############################
    def get_physical_Loss_original_dim_fft(self, output):
        start=0
        k2=self.prepar_fft()
        #print(k2.shape)  
        phi = output[start:, 0:1, :, :]
        lent = phi.shape[0]
        lenx = phi.shape[3]
        leny = phi.shape[2]
        len_ch=phi.shape[1]
        phi_conv1d = phi.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        phi_conv1d = phi_conv1d.reshape(lenx*leny,len_ch,lent)
        phi_t = self.dt_op_dim(phi_conv1d)  # lent ==> lent-2 due to no-padding
        phi_t = phi_t.reshape(leny, lenx, 1, lent)
        phi_t = phi_t.permute(3, 2, 0, 1)  # [step, c, height(Y), width(X)]   
        phi = output[start:, 0:1, :, :]  # [t, c, height(Y), width(X)]
        #phi=torch.clamp(phi.clone(), min=0.0 , max=1.0)
        phi_k = torch.fft.fftn(phi, dim=[2, 3])
        #print("phi_k.shape:",phi_k.shape)
        phi_t_k = torch.fft.fftn(phi_t, dim=[2, 3])
        #print("phi_t_k.shape:",phi_t_k.shape)

        k2_expanded = k2.unsqueeze(0).unsqueeze(0)
        
        # Compute the term inside the square brackets
        right_side_eqn_k = -k2_expanded*phi_k + (phi_k - 0.5)

        # Compute f_phi_k
        f_phi_k =  phi_t_k- right_side_eqn_k #

        return f_phi_k
    ###############################
    def generate_collocation_points(self,batch_XYT, batch_phi, x_max, x_min, y_max, y_min, t_current, number_points_per_batch=10):
        t_min=t_current
        t_max=t_min + self.dt 
        R0 = self.eta/3
        
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
                collocation_t =  random.uniform(t_min, t_max) 

                batch_X_f.append([collocation_x, collocation_y, collocation_t])

        batch_X_f = np.array(batch_X_f)
        return batch_X_f   
    ###############################
    def compute_gradients(self, x, y, t):  
        x = x.clone().detach().requires_grad_(True)  # Ensure x requires gradients
        y = y.clone().detach().requires_grad_(True)  # Ensure y requires gradients
        t = t.clone().detach().requires_grad_(True)  # Ensure t requires gradients
        
        g = torch.cat((x, y, t), dim=1)
        phi = self.model.ANN(g)

        # Compute gradients
        phi_t = torch.autograd.grad(phi, t, torch.ones_like(phi), create_graph=True)[0]
        phi_x = torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=True)[0]
        phi_y = torch.autograd.grad(phi, y, torch.ones_like(phi), create_graph=True)[0]

        phi_xx = torch.autograd.grad(phi_x, x, torch.ones_like(phi_x), create_graph=True)[0]
        phi_yy = torch.autograd.grad(phi_y, y, torch.ones_like(phi_y), create_graph=True)[0]

        lap_phi = phi_xx + phi_yy

        return phi, phi_t, lap_phi
    ###############################
    def get_physical_Loss_original_dim_int(self, phi):
        #tf.print("phi",phi.shape)
        lb = np.array([0, 0,0])
        ub = np.array([self.Nx*self.dx, self.Ny*self.dy,self.Nt*self.dt]) 
        x = np.linspace(lb[0], ub[0], self.Nx)
        y = np.linspace(lb[1], ub[1], self.Ny)
        X, Y = np.meshgrid(x, y)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        
        cumulative_loss = torch.tensor(0.0, requires_grad=True) 
        resample_ratio = 0.75  
        resample_ratio_grain=0.1
        
        phi_seq = phi[0, 0]
        
        for i in range(0,phi.shape[0]-1):  
            #phi_seq = phi[i, 0]#.detach().numpy()            
            phi_flattened_np = phi_seq.flatten()

            interfacial_indices = np.where(np.logical_and(phi_flattened_np > 0.05, phi_flattened_np < 0.95))[0]
            non_interfacial_indices = np.where(np.logical_or(phi_flattened_np <= 0.05, phi_flattened_np >= 0.95))[0]
            
            num_samples_interfacial = int(len(interfacial_indices) * resample_ratio)
            sampled_indices_interfacial = np.random.choice(interfacial_indices, num_samples_interfacial, replace=False)
            
            num_samples_non_interfacial = int(len(non_interfacial_indices) * resample_ratio_grain)
            sampled_indices_non_interfacial = np.random.choice(non_interfacial_indices, num_samples_non_interfacial, replace=False)

            sampled_indices =sampled_indices_interfacial#np.concatenate((sampled_indices_interfacial, sampled_indices_non_interfacial), axis=0)
            
            interfacial_x = X.flatten()[sampled_indices]
            interfacial_y = Y.flatten()[sampled_indices]
                    
            t_current = i * self.dt
            
            time_vector = np.full_like(interfacial_x, t_current)
            X_Y_T = np.column_stack((interfacial_x, interfacial_y, time_vector))
            interfacial_phi = phi_flattened_np[sampled_indices]
            
            collocation_points = self.generate_collocation_points(
                batch_XYT=X_Y_T,
                batch_phi=interfacial_phi,
                x_max=ub[0],
                x_min=lb[0],
                y_max=ub[1],
                y_min=lb[1],
                t_current=t_current,
                number_points_per_batch=10
            )
            
            t_seq = torch.tensor(collocation_points[:, 2:3], dtype=self.dtype, requires_grad=True)
            x_seq = torch.tensor(collocation_points[:, 0:1], dtype=self.dtype, requires_grad=True)
            y_seq = torch.tensor(collocation_points[:, 1:2], dtype=self.dtype, requires_grad=True)
            phi_pred, phi_t, lap_phi = self.compute_gradients(x_seq, y_seq, t_seq)
                
            if isinstance(phi_pred, torch.Tensor):
                phi_pred_np = phi_pred.detach().numpy() 
            else: 
                phi_pred_np =  phi_pred
                
            phi_term = (np.pi**2 / self.eta**2) * (phi_pred - 0.5)
            right_side_eqn = self.mu * self.sigma * (lap_phi + phi_term) #+ self.delta_g * self.h(phi_flattened))
            f_phi = torch.tensor(phi_t - right_side_eqn, dtype=self.dtype)        
            
            time = np.full_like(x, t_current)
            time_flat = np.repeat(time[:, np.newaxis], self.Ny, axis=1).flatten()
            g = np.stack((X_flat, Y_flat, time_flat), axis=1)
            g = torch.tensor(g, dtype=self.dtype)
            phi_2d = self.model.ANN(g).detach().numpy().reshape(self.Nx, self.Ny)
            phi_seq= phi_2d
            loss_sequence =  mse_loss(f_phi, torch.zeros_like(f_phi)) 

            cumulative_loss = cumulative_loss +loss_sequence
        
        return cumulative_loss
    ##########################################################
    ###########################    
    def compute_laplacian(self,img, ksize, delta_x=1.0, delta_y=1.0):
        """
        Compute the Laplacian of an image using OpenCV's separable filters,
        scaled by the grid spacings delta_x and delta_y.
        """
        deriv_kx2, _ = cv2.getDerivKernels(dx=2, dy=0, ksize=ksize, normalize=True)
        _, deriv_ky2 = cv2.getDerivKernels(dx=0, dy=2, ksize=ksize, normalize=True)
        
        deriv_kx2 = np.divide(deriv_kx2, delta_x**2)  
        deriv_ky2 = np.divide(deriv_ky2, delta_y**2)  
        
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        grad_x2 = cv2.sepFilter2D(img, cv2.CV_64F, deriv_kx2, np.array([[1]], dtype=np.float32))
        grad_y2 = cv2.sepFilter2D(img, cv2.CV_64F, np.array([[1]], dtype=np.float32), deriv_ky2)
        
        laplacian = grad_x2 + grad_y2
        return laplacian            
        
    ###########################    
    def compute_laplacian_batch(self,batch, ksize=5, delta_x=1.0, delta_y=1.0):
        """
        Compute the Laplacian for a batch of images.
        """
        batch_size, channels, height, width = batch.shape
        laplacian_batch = torch.empty_like(batch)
    
        for i in range(batch_size):
            for j in range(channels):
                img = batch[i, j].cpu().detach().numpy()
                laplacian = self.compute_laplacian(img, ksize, delta_x, delta_y)
                laplacian_batch[i, j] = torch.tensor(laplacian, dtype=self.dtype)
    
        return laplacian_batch   
    ###########################

    ###############################
    def get_physical_Loss_original_dim(self, output,get_loss_pde=True,  get_loss_energy=False,check_residual=False):
        
        if len(output)>3:
            get_loss_energy=True
            
        # worker function for f_phi
        f_phi = None
        f_phi_2 = None
        max_res = None
        F = None
        dF_dt = None
        RHS = None
        out_t = None
        f_E_phi= None
        ###########################
        def compute_f_phi():
            nonlocal f_phi,f_phi_2
            f_phi = self.compute_residual(phi, self.dt, self.mu, self.sigma, self.eta, self.delta_g)
        
        #  worker for F
        # ###########################
        def compute_F():
            nonlocal f_E_phi, F, dF_dt, RHS, out_t
            f_E_phi, F, dF_dt, RHS, out_t = self.get_energy_terms(phi) 
            
        ###########################    
        def compute_laplacian_base(phi, dx, dy):
            # Use torch.roll to shift the tensor along specified dimensions
            laplacian = (torch.roll(phi, shifts=1, dims=2) + torch.roll(phi, shifts=-1, dims=2) +
                        torch.roll(phi, shifts=1, dims=3) + torch.roll(phi, shifts=-1, dims=3) - 4 * phi) / (dx * dy)
            return laplacian

        # shape: [t, c, h, w]
        """
        # ----------Padding x axis due to periodic boundary condition ------
        The last two columns of output output[:, :, :, -2:]
        are concatenated with the beginning of output, 
        and the first three columns of output are concatenated 
        with the end of output. This ensures that the periodic boundary 
        conditions are considered along the x-axis.
        
        """
        start=0
        end=0     
        max_ch=output.shape[1]
        phi= output[:, 0:max_ch, ::]

        #
        ###########################
        ###########################
        ###########################
        # loss Energy
        if get_loss_energy:
            f_E_phi,F,dF_dt, RHS,out_t =self.get_energy_terms(phi)
        
        thread_f_phi = threading.Thread(target=compute_f_phi)
        thread_F = threading.Thread(target=compute_F)

        if get_loss_pde:
            thread_f_phi.start()

        if get_loss_energy:
            thread_F.start()

        # Wait for threads to 
        if get_loss_pde:
            thread_f_phi.join()
        if get_loss_energy:
            thread_F.join()
                
        if check_residual:  # using broadcasting to check residual (max_res 1 should be equal to max_res 2)
            #laplacian_np = compute_laplacian_base(phi, self.dx,self.dy).detach().numpy()
            phi_np=phi.detach().numpy()
            phi_t_np=(phi_np[1:] - phi_np[:-1]) / self.dt
            laplacian_np = (np.roll(phi_np, shift=1, axis=2) + np.roll(phi_np, shift=-1, axis=2) +
            np.roll(phi_np, shift=1, axis=3) + np.roll(phi_np, shift=-1, axis=3) - 
             4 * phi_np) / (self.dx * self.dy)
            right_side_eqn_np = self.mu * (self.sigma * (laplacian_np + (np.pi**2 / (2 * self.eta**2)) * (2 * phi.cpu().detach().numpy() - 1)))

            right_side_eqn_np=right_side_eqn_np[1:] 
            
            residual = phi_t_np- right_side_eqn_np# np.where((phi_new_unclipped < 0) | (phi_new_unclipped > 1), 0, phi_t.cpu().detach().numpy() - right_side_eqn_np)
            max_res = np.max(residual)
            mse_f_phi = np.mean(np.square(residual - np.zeros_like(residual)))

            
        if check_residual:
            max_res = 0
            for t in range(1,phi.shape[0] - 1): 
            
                phi_current = phi[t].cpu().unsqueeze(0).detach().numpy()
                phi_prev = phi[t-1].cpu().unsqueeze(0).detach().numpy()
                laplacian = (np.roll(phi_prev, shift=1, axis=2) + np.roll(phi_prev, shift=-1, axis=2) +
                np.roll(phi_prev, shift=1, axis=3) + np.roll(phi_prev, shift=-1, axis=3) - 
                4 * phi_prev) / (self.dx * self.dy)     

                phi_new = phi_prev + self.dt * self.mu * self.sigma * (laplacian + np.pi**2 / (2 * self.eta**2) * (2 * phi_prev - 1))
                phi_t = (phi_new- phi_prev) / self.dt           
                        
                right_side_eqn = self.mu * self.sigma * (laplacian + np.pi**2 / (2 * self.eta**2) * (2 * phi_prev - 1))
                
                residual = phi_t- right_side_eqn
                
                phi_t_min = phi_t.min()
                phi_t_max = phi_t.max()
                right_side_eqn_min = right_side_eqn.min()
                right_side_eqn_max = right_side_eqn.max()
                
                residual_min = residual.min()
                residual_max = residual.max()
                if abs(residual_max) > max_res:
                    max_res = abs(residual_max)
                
                if t % 100 == 0:  
                    tf.print(f"Time step = {t}: residual min = {residual_min}, residual max = {residual_max}")
            
            
        if check_residual:
            self.plot_residuals(F,dF_dt, RHS,out_t)
        
        if get_loss_energy==False:
            f_E_phi=torch.zeros_like(f_phi)
            
        if get_loss_pde==False:
            f_phi_2=torch.zeros_like(f_E_phi)
            f_phi=torch.zeros_like(f_E_phi)

        return f_phi,f_E_phi#  torch.zeros_like(f_phi)  # f_phi  f_E_phi
    ##########################################################
    ###############################################
    def get_physical_Loss_latent(self,output_latent):

        def compute_laplacian(phi, dx, dy):
            laplacian = (torch.roll(phi, shifts=1, dims=2) + torch.roll(phi, shifts=-1, dims=2) +
                        torch.roll(phi, shifts=1, dims=3) + torch.roll(phi, shifts=-1, dims=3) - 4 * phi) / (dx * dy)
            return laplacian   
        start=0
        end=-1
        max_ch=output_latent.shape[1]
        phi=output_latent[:, start:max_ch, ::]

        laplace_phi= (
            torch.roll(phi, shifts=1, dims=2) +
            torch.roll(phi, shifts=-1, dims=2) +
            torch.roll(phi, shifts=1, dims=3) +
            torch.roll(phi, shifts=-1, dims=3) -
            4 * phi
        ) / (self.dx * self.dy) 

        
        laplace_phi= laplace_phi[:-1]  
        phi_t = (phi[1:] - phi[:-1]) / self.dt
        phi = phi[:-1]
        
        assert laplace_phi.shape == phi_t.shape
        assert laplace_phi.shape == phi.shape
        
        # loss PDE 
        mu=self.mu
        sigma=self.sigma
        eta=self.eta
        delta_g=self.delta_g
        phi_term = (np.pi**2 /eta**2  ) * (phi - 1/2) 
        right_side_eqn =  mu*  (sigma * (laplace_phi+ phi_term )  ) # 
        f_phi = phi_t -right_side_eqn

        return f_phi
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    def compute_loss(self, output,output_latent, outputs_ann_current):
        
        def mse_loss(predicted, target):
            return F.mse_loss(predicted, target)
        
        import torch.nn.functional as F
        ''' compute the phycisal loss '''
        global flag_scipy
        
        # compute te loss on the latent dimension
        if self.opt=="latent":  # the loss is computed on the latent dimension 
            f_phi  = self.get_physical_Loss_latent(output_latent)
            loss_phi = mse_loss(f_phi, torch.zeros_like(f_phi)) 
            return loss_phi

        else:
            f_phi_lstm, f_E_phi_lstm= self.get_physical_Loss_original_dim(output)
            
            f_phi_ann, f_E_phi_ann= self.get_physical_Loss_original_dim(outputs_ann_current)

        loss_phi_lstm = mse_loss(f_phi_lstm, torch.zeros_like(f_phi_lstm)) 
        
        loss_phi_ann= torch.zeros_like(f_phi_lstm)#


        loss_diff = mse_loss(output[1:], outputs_ann_current[1:]) if output.shape == outputs_ann_current.shape else torch.zeros_like(loss_phi_lstm)

        loss_E_phi=   mse_loss(f_E_phi_lstm, torch.zeros_like(f_E_phi_lstm)) 

        return  loss_phi_lstm , loss_E_phi, loss_diff  
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    def compute_residual(self, phi, dt, mu, sigma, eta, delta_g):
        if len(phi)>5:
            phi=phi[1:] 

        ###############################
        ########### Conv deriv ########
        ###############################
        
        ###############################
        ########## Roll or csv Method  
        ###############################        
        #"""
        phi_t = (phi[1:] - phi[:-1]) / self.dt
        
        laplace_phi = (torch.roll(phi, shifts=1, dims=2) + torch.roll(phi, shifts=-1, dims=2) +
                    torch.roll(phi, shifts=1, dims=3) + torch.roll(phi, shifts=-1, dims=3) -
                    4 * phi) / (self.dx * self.dy)
        #laplace_phi=self.compute_laplacian_batch(phi, ksize=5, delta_x=self.dx, delta_y=self.dx)

        right_side_eqn = mu * (sigma * (laplace_phi + torch.pi**2 / (2 * eta**2) * (2 * phi - 1)))  # +h_term(phi, eta) * delta_g if applicable
        right_side_eqn=right_side_eqn[:-1]
        
        assert laplace_phi.shape == phi.shape
        assert right_side_eqn.shape == phi_t.shape  
            
        f_phi= (phi_t - right_side_eqn)

        #"""

        return f_phi

    ###############################
    def get_energy_terms(self, Phi):
        Phi = Phi.clone().detach().requires_grad_(True).float()
        
        def get_free_energy(eta, phi, dphi_dt, sigma, delta_g, dx, dy):
            dphi_dx = (torch.roll(phi, shifts=-1, dims=1) - torch.roll(phi, shifts=1, dims=1)) / (2 * dx)
            dphi_dy = (torch.roll(phi, shifts=-1, dims=0) - torch.roll(phi, shifts=1, dims=0)) / (2 * dy)
            
            dphi_dt_dx = (torch.roll(dphi_dt, shifts=-1, dims=1) - torch.roll(dphi_dt, shifts=1, dims=1)) / (2 * dx)
            dphi_dt_dy = (torch.roll(dphi_dt, shifts=-1, dims=0) - torch.roll(dphi_dt, shifts=1, dims=0)) / (2 * dy)
            
            dot_product = dphi_dx * dphi_dt_dx + dphi_dy * dphi_dt_dy
            
            term1 = (4 * sigma / eta) * (dphi_dt * (1 - 2 * phi))
            term2 = (4 * sigma / eta) * ((2 * eta**2 / torch.pi**2) * dot_product)
            
            F = torch.sum(term1 + term2) * dx * dy  # Over the domain
            return F

        Phi_t = (Phi[1:] - Phi[:-1]) / self.dt  # Time derivative of Phi
        
        L = np.pi**2 * self.mu / (8 * self.eta)

        out_energy_ = []
        out_energy_deriv_ = []
        out_rhs_ = []
        out_time_ = []

        for istep in range(1, len(Phi)):
            phi_prev = Phi[istep - 1].unsqueeze(0)
            
            laplacian = (torch.roll(phi_prev, shifts=1, dims=2) + torch.roll(phi_prev, shifts=-1, dims=2) +
                        torch.roll(phi_prev, shifts=1, dims=3) + torch.roll(phi_prev, shifts=-1, dims=3) - 
                        4 * phi_prev) / (self.dx * self.dy)

            right_side_eqn = self.mu * self.sigma * \
                            (laplacian + (torch.pi**2 / (2 * self.eta**2)) * (2 * phi_prev - 1))
                            
            phi_t = Phi_t[istep - 1]  # Correctly index into Phi_t
            t = istep * self.dt  # Current time

            F = get_free_energy(self.eta, phi_prev, phi_t, self.sigma, self.delta_g, self.dx, self.dy)
            out_energy_.append(F)

            if istep > 1:  # Starting from the second step
                out_time_.append(t)
                dF_dt = (out_energy_[-1] - out_energy_[-2]) / self.dt
                phi_t_squared_integral = torch.sum(phi_t**2) * self.dx * self.dy
                rhs = - (1 / L) * phi_t_squared_integral

                out_energy_deriv_.append(dF_dt)
                out_rhs_.append(rhs)

        out_energy_ = torch.stack(out_energy_)
        out_energy_deriv_ = torch.stack(out_energy_deriv_)
        out_rhs_ = torch.stack(out_rhs_)
        out_time_ = torch.tensor(out_time_, dtype=self.dtype)

        f_E_phi = out_energy_deriv_ - out_rhs_ # Compute difference using tensors

        return f_E_phi, F, out_energy_deriv_, out_rhs_, out_time_

    ###############################
    def plot_residuals(self,F,out_energy_deriv, out_rhs,out_time):
        out_energy_deriv= np.asarray(out_energy_deriv)
        out_rhs = np.asarray(out_rhs)
        energy_residual = out_energy_deriv-out_rhs

        plt.figure(figsize=(12, 6))

        plt.plot(out_time, out_energy_deriv, "r--", label="Energy residual")

        plt.plot(out_time, out_rhs, "b--", label="Energy derivative")

        plt.plot(out_time, energy_residual, "m--", label="RHS")

        residual = np.abs(energy_residual)
        max_residual = np.max(residual)
        min_residual = np.min(residual)

        plt.axhline(y=max_residual, color="y", linestyle="--", label="Max residual")
        plt.axhline(y=min_residual, color="c", linestyle="--", label="Min residual")

        plt.text(out_time[-1], max_residual, f"Max: {max_residual:.2e}", color="b", ha="right", va="bottom")
        plt.text(out_time[-1], min_residual, f"Min: {min_residual:.2e}", color="g", ha="right", va="top")

        plt.legend()
        plt.title("Energy Residual, Energy Derivative, and RHS")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.grid(True)
        figures_directory = 'figures'
        filename = f"Energy_Residual.png"
        filepath = os.path.join(figures_directory, filename)
        plt.savefig(filepath)
        plt.close()   
    ###############################
def mse_loss(predicted, target):
    return F.mse_loss(predicted, target) 
