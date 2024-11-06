# %%
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

from importlib import reload
#from sklearn.preprocessing import MinMaxScaler

import L_PINN  # python files (classes)
import pre_post
from pre_post import *
from L_PINN import *


# %%
def generate_circles(mean_r, num_circles, std, Nx, Ny, Nz):
    # Initialize the arrays for the radii and centers of the circles
    R0 = np.zeros(num_circles)
    X_center = np.zeros(num_circles)
    Y_center = np.zeros(num_circles)
    Z_center = np.zeros(num_circles)

    # Generate the first circle randomly
    R0[0] = np.random.normal(loc=mean_r, scale=std)
    X_center[0] = np.random.randint(R0[0], Nx-R0[0])
    Y_center[0] = np.random.randint(R0[0], Ny-R0[0])
    Z_center[0] = np.random.randint(R0[0], Nz-R0[0])

    # Loop through the remaining circles and generate them one at a time
    for i in range(1, num_circles):
        # Flag to indicate whether the new circle overlaps with any existing circles
        overlaps = True
        while overlaps:
            # Generate the radius and center of the new circle randomly
            R0[i] = np.random.normal(loc=mean_r, scale=std)
            X_center[i] = np.random.randint(R0[i], Nx-R0[i])
            Y_center[i] = np.random.randint(R0[i], Ny-R0[i])
            Z_center[i] = np.random.randint(R0[i], Nz-R0[i])

            # Check the new circle against the existing circles
            overlaps = False
            for j in range(i):
                if np.sqrt((X_center[i]-X_center[j])**2 + (Y_center[i]-Y_center[j])**2 ) < (R0[i]+R0[j]): #+ (Z_center[i]-Z_center[j])**2
                    overlaps = True
                    break
    
    return R0, X_center, Y_center, Z_center

# %%
"""
#### Model paramters
"""

# %%


# %%
   # Grid parameters
Nx=64
Ny=64
Nt=1500 
lb = np.array([0, 0,0])
ub = np.array([1, 1,1500]) 
dx = (ub[0] - lb[0]) / (Nx - 1)
dy = (ub[1] - lb[1]) / (Ny - 1)
# physical parameters
num_phases=1
dt = (ub[2] - lb[2]) / (Nt - 1)
sigma=1
mu=5e-5
delta_g= 0 
eta=7*dx

x = np.linspace(lb[0], ub[0], Nx)
y = np.linspace(lb[1], ub[1], Ny)
t= np.linspace(lb[2], ub[2], Nt) 
X, Y, T = np.meshgrid(np.linspace(lb[0], ub[0], Nx),
                        np.linspace(lb[1], ub[1], Ny),
                        np.linspace(lb[2], ub[2], Nt),
                        indexing='ij')
#### Conv-LSTM hyper-parameters
input_channels=num_phases
hidden_channels = [8, 32, 64, 64]  #  number of output channels for each layer in the network.
input_kernel_size = [3, 3, 3, 3] #    kernel sizes for 3 auto-encoders and 1 conv-lstm : (4,4), (4,4), (4,4) and (3,3)
input_stride = [2, 2, 2, 1]  #  for each auto-encoder, the stride is (2,2) then it is (1,1) for the conv-lstm
input_padding = [1, 1, 1, 1]  # a standard minimum, padding of (1,1) for each architecture 
num_layers = [3, 1]  # 3 auto-enocders + 1 conv-lstm 

inter_dim=512
latent_dim=128

# %%
save_path="figures"
import pre_post

from pre_post import *
from L_PINN import *
# %%
Pre_Post=PrePost(X=X,T=None, lb=lb, ub=ub, Nx=Nx,Ny=Ny,dx=dx,dy=dy,x=x,y=y, eta=eta,\
        phi_true=None)
Pre_Post.EraseFile(path=os.path.join(os.getcwd(),"VAE_figs")) 

# %%
"""
### Generate a Dataset of Initializations
"""



# Define the range of radii and eta
radii_range = np.arange(25, 26.5, 0.05)* dx
eta_range = np.arange(7, 7.001, 0.05)* dx

# Generate datasets for different radii and etas
datasets = []

for radius in radii_range:
        radius = np.array([radius])  # Convert radius to numpy array
        for eta in eta_range:
                #print(radius,eta)
                R0, X_center, Y_center, Z_center = generate_circles(mean_r=radius, num_circles=1, std=0, Nx=Nx, Ny=Ny, Nz=100)
                
                X_center = np.array([Nx * dx / 2])
                Y_center = np.array([Ny * dy / 2])
                #print(R0, X_center,Y_center)
                #stop
                phi_0, X_ini_all = Pre_Post.init_micro_cir(R0, eta, X_center, Y_center, Z_center, Nx, Ny, x, y)
                
                phi_0 = torch.tensor(phi_0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                #print(phi_0.shape)
        
                # Add initial conditions to dataset
                datasets.append(phi_0)
tf.print("len(datasets)",len(datasets))
#"""

# %%
datasets_dict = {'dataset_{}'.format(i): dataset for i, dataset in enumerate(datasets)}
np.savez('datasets.npz', **datasets_dict)


# %%
#"""
loaded_data = np.load('datasets.npz')
datasets = [loaded_data[key] for key in loaded_data.keys()]
datasets = [torch.tensor(dataset, dtype=torch.float32) for dataset in datasets]
datasets = torch.stack(datasets, dim=0)
tf.print("datasets" ,datasets.shape)
#"""

# %%
"""
datasets = [torch.tensor(dataset, dtype=torch.float32) for dataset in datasets]
datasets = torch.stack(datasets, dim=0)
print(datasets.shape)
IC_tensor = datasets[39000,...]
print(datasets.shape, IC_tensor.shape)
# Create a heatmap for the initial condition
plt.imshow(IC_tensor[0,0].clone().detach(), cmap='viridis', origin='lower')
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$\phi_0$')
plt.savefig("IC_whole_solution.png")
plt.show()
"""

# %%
#"""
import numpy as np

# Define the number of rows and columns for the subplot grid
N = 3  # Number of rows
M = 4  # Number of columns

# Create a figure and axis for the subplot grid
fig, axes = plt.subplots(N, M, figsize=(12, 10))

# Generate random indices for selecting samples
num_samples = N * M
sample_indices = np.random.choice(len(datasets), num_samples, replace=False)

# Iterate over the random sample indices and plot images
for idx, ax in zip(sample_indices, axes.ravel()):
    # Extract the image from the dataset
    uv_ic_channel_0 = datasets[idx][0, 0, :, :]
    
    # Plot the image on the corresponding subplot
    ax.imshow(uv_ic_channel_0, cmap='viridis', origin='lower', extent=[0, uv_ic_channel_0.shape[1], 0, uv_ic_channel_0.shape[0]])
    ax.set_title(f'Sample {idx}')  # Set the title for the subplot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('off')  # Turn off axis

# Add a colorbar to the last subplot
#cbar = fig.colorbar(axes[0, 0].imshow(uv_ic_channel_0, cmap='viridis', origin='lower', extent=[0, uv_ic_channel_0.shape[1], 0, uv_ic_channel_0.shape[0]]), ax=axes.ravel().tolist(), shrink=0.95)
#cbar.set_label('u')  # Set the label for the colorbar

# Adjust layout
plt.tight_layout()
save_dir= os.path.join(os.getcwd(), "VAE_figs")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir,"sample_Dataset"))
plt.close()
#"""

# %%
"""
import pre_post
from pre_post import *
reload(pre_post)
Pre_Post=PrePost(X=X,T=None, lb=lb, ub=ub, Nx=Nx,Ny=Ny,dx=dx,dy=dy,x=x,y=y, eta=eta,\
        phi_true=None)
"""
R0, X_center, Y_center,Z_center =\
        generate_circles(mean_r=25*dx,num_circles=1, std=0, Nx=Nx, Ny=Ny,Nz=100) #25*dx
X_center=np.array([Nx*dx/2])
Y_center=np.array([Ny*dx/2])

R0=25*dx
phi_0, X_ini_all=Pre_Post.init_micro_cir(R0,eta, X_center,Y_center, Z_center,Nx,Ny,x,y) 

phi_0 = torch.tensor(phi_0, dtype=torch.float32)
save_dir= os.path.join(os.getcwd(), "VAE_figs")
plt.savefig(os.path.join(save_dir,"phi0"))
plt.close()


# %%
# %%
"""
T_ch=eta**2 /(sigma*mu)
x=x/eta
y=y/eta
t=t/T_ch 
dt =dt /T_ch 
dx=dx/eta
dy=dy/eta
R0=R0/eta
delta_g=eta*delta_g
lb[0], ub[0]=lb[0]/eta, ub[0]/eta
lb[1], ub[1]= lb[1]/eta, ub[1]/eta
lb[2], ub[2]= lb[2]/T_ch, ub[2]/T_ch
ub=ub/eta
"""


# %%
"""
#### define the variational autoencoder 
"""

# %%
def compute_output_shape(W_in, kernel_size, stride, padding):
    W_out = (W_in - kernel_size + 2 * padding) // stride + 1
    return W_out

#  input dimensions
W_in = 64

# Encoder block 0
encoder0_kernel_size =3
encoder0_stride = 2
encoder0_padding = 1

encoder_params = [
    {"kernel_size":3, "stride": 2, "padding": 1},  # Encoder block 0 parameters
    {"kernel_size": 3, "stride": 2, "padding": 1},# Encoder block 1 parameters
    {"kernel_size": 3, "stride": 2, "padding": 1}   # Encoder block 1 parameters
]


# Iterate through encoder blocks and compute output shapes
for i, params in enumerate(encoder_params):
    W_out = compute_output_shape(W_in, params["kernel_size"], params["stride"], params["padding"])
    #tf.print("Output shape after encoder block {}:".format(i), W_out)
    
    # Update input shape for the next encoder block
    W_in = W_out

# %%
#"""
import L_PINN
reload(L_PINN)  # mandatory to reload content at each re-call atfer modification
from L_PINN import *

input  = phi_0.clone().detach().to(torch.float32).unsqueeze(0).unsqueeze(0)
VAE = VariationaAutoencoder( input_tensor=input,   input_channels = num_phases, # one phase 
    hidden_channels = hidden_channels,  #  number of output channels for each layer in the network.
    input_kernel_size = input_kernel_size, #    kernel sizes for 3 auto-encoders and 1 conv-lstm : (4,4), (4,4), (4,4) and (3,3)
    input_stride = input_stride,  #  for each auto-encoder, the stride is (2,2) then it is (1,1) for the conv-lstm
    input_padding = input_padding,  # a standard minimum, padding of (1,1) for each architecture 
    num_layers = num_layers,  # 3 auto-enocders + 1 conv-lstm 
    inter_dim=inter_dim,
    latent_dim=latent_dim,
    )


# %%
"""
#### train the variational autoencoder 
"""

# %%
import L_PINN
reload(L_PINN)  # mandatory to reload content at each re-call atfer modification
from L_PINN import *
#datasets=input.unsqueeze(0)
dataloader = DataLoader(datasets, batch_size=128, shuffle=True)
VAE.train(dataloader,save_path="VAE_figs",datasets=datasets)

# %%
VAE.check_IC(datasets,save_dir)


