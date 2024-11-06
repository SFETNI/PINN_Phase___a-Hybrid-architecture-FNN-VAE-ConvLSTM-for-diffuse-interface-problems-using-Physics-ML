import torch

# Initialize weights and alphas
self.ic_weight = 0.5
self.motion_weight = 1.0
self.energ_weight = 0.5
self.data_weight = 0.0

alpha_ic = torch.tensor(1.0, requires_grad=True)
alpha_motion = torch.tensor(1.0, requires_grad=True)
alpha_energ = torch.tensor(1.0, requires_grad=True)
alpha_data = torch.tensor(1.0, requires_grad=True)

def compute_ann_loss():
    # Compute individual losses
    loss_IC_ann = ...
    grid_motion_loss = ...
    loss_E_phi = ...
    data_loss = mse_loss(phi2D_ann, Grund_truth[-1].unsqueeze(0))
    
    # Compute total weighted loss
    total_loss = alpha_ic * loss_IC_ann + alpha_motion * grid_motion_loss + alpha_energ * loss_E_phi + alpha_data * data_loss
    return total_loss, loss_IC_ann, grid_motion_loss, loss_E_phi, data_loss

for epoch in range(num_epochs):
    # Compute losses
    total_loss, loss_IC_ann, grid_motion_loss, loss_E_phi, data_loss = compute_ann_loss()
    
    # Backpropagation
    total_loss.backward()
    
    # Compute gradient norms
    G_ic = torch.norm(torch.autograd.grad(loss_IC_ann, model.parameters(), retain_graph=True, create_graph=True))
    G_motion = torch.norm(torch.autograd.grad(grid_motion_loss, model.parameters(), retain_graph=True, create_graph=True))
    G_energ = torch.norm(torch.autograd.grad(loss_E_phi, model.parameters(), retain_graph=True, create_graph=True))
    G_data = torch.norm(torch.autograd.grad(data_loss, model.parameters(), retain_graph=True, create_graph=True))
    
    # Compute average gradient norm
    G_avg = (G_ic + G_motion + G_energ + G_data) / 4
    
    # Update alphas
    alpha_ic = alpha_ic * (G_avg / G_ic).detach()
    alpha_motion = alpha_motion * (G_avg / G_motion).detach()
    alpha_energ = alpha_energ * (G_avg / G_energ).detach()
    alpha_data = alpha_data * (G_avg / G_data).detach()
    
    # Normalize alphas
    total_weight = alpha_ic + alpha_motion + alpha_energ + alpha_data
    alpha_ic /= total_weight
    alpha_motion /= total_weight
    alpha_energ /= total_weight
    alpha_data /= total_weight
    
    # Proceed with optimizer step
    optimizer.step()
    optimizer.zero_grad()





import torch
from scipy.optimize import minimize

# Initialize weights and alphas
self.ic_weight = 0.5
self.motion_weight = 1.0
self.energ_weight = 0.5
self.data_weight = 0.0

alpha_ic = torch.tensor(1.0, requires_grad=True)
alpha_motion = torch.tensor(1.0, requires_grad=True)
alpha_energ = torch.tensor(1.0, requires_grad=True)
alpha_data = torch.tensor(1.0, requires_grad=True)

def compute_ann_loss():
    # Compute individual losses
    loss_IC_ann = ...
    grid_motion_loss = ...
    loss_E_phi = ...
    data_loss = mse_loss(phi2D_ann, Grund_truth[-1].unsqueeze(0))
    
    # Compute total weighted loss
    total_loss = alpha_ic * loss_IC_ann + alpha_motion * grid_motion_loss + alpha_energ * loss_E_phi + alpha_data * data_loss
    return total_loss, loss_IC_ann, grid_motion_loss, loss_E_phi, data_loss

def closure():
    optimizer.zero_grad()
    
    # Compute losses
    total_loss, loss_IC_ann, grid_motion_loss, loss_E_phi, data_loss = compute_ann_loss()
    
    # Backpropagation
    total_loss.backward()
    
    # Compute gradient norms
    G_ic = torch.norm(torch.autograd.grad(loss_IC_ann, model.parameters(), retain_graph=True, create_graph=True))
    G_motion = torch.norm(torch.autograd.grad(grid_motion_loss, model.parameters(), retain_graph=True, create_graph=True))
    G_energ = torch.norm(torch.autograd.grad(loss_E_phi, model.parameters(), retain_graph=True, create_graph=True))
    G_data = torch.norm(torch.autograd.grad(data_loss, model.parameters(), retain_graph=True, create_graph=True))
    
    # Compute average gradient norm
    G_avg = (G_ic + G_motion + G_energ + G_data) / 4
    
    # Update alphas
    alpha_ic = alpha_ic * (G_avg / G_ic).detach()
    alpha_motion = alpha_motion * (G_avg / G_motion).detach()
    alpha_energ = alpha_energ * (G_avg / G_energ).detach()
    alpha_data = alpha_data * (G_avg / G_data).detach()
    
    # Normalize alphas
    total_weight = alpha_ic + alpha_motion + alpha_energ + alpha_data
    alpha_ic /= total_weight
    alpha_motion /= total_weight
    alpha_energ /= total_weight
    alpha_data /= total_weight
    
    return total_loss.item()

# Example usage with SciPy optimizer
result = minimize(closure, initial_params, method='L-BFGS-B', jac=True)
