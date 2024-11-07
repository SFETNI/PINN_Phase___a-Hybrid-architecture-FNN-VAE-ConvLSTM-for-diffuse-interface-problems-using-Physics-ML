import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
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
import math
import torch.nn.init as init

class SpatialCorrelation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SpatialCorrelation, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)
        self.out_features = out_features

    def forward(self, xi, xj):
        #print("in SpatialCorrelation" , xi.shape, xj.shape)
        batch_size=1
        batch_size,ch, xL, yL = xi.size()
        in_features = xi.view(batch_size, -1).size(1)
        self.fc1 = nn.Linear(in_features, self.out_features)
        self.fc2 = nn.Linear(in_features, self.out_features)

        xi_flattened = xi.view(batch_size, -1)
        xj_flattened = xj.view(batch_size, -1)
        #print("xi_flattened", xi_flattened.shape)

        # Pass through linear layers
        xi_fc = F.relu(self.fc1(xi_flattened))
        xj_fc = F.relu(self.fc2(xj_flattened))
        
        # reshape flattened outputs to match original input shape
        xi_fc = xi_fc.view(batch_size, ch, xL, yL)
        xj_fc = xj_fc.view(batch_size, ch, xL, yL)

        rij = torch.mul(xi_fc, xj_fc)  # Element-wise multiplication
        rji = torch.mul(xj_fc, xi_fc)  # Element-wise multiplication
        
        return rij, rji
########################################
class SelfInformationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfInformationNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x
    
########################################
class MultiHeadSpatialAttention(nn.Module):
    def __init__(self, input_dim, num_heads=64):
        super(MultiHeadSpatialAttention, self).__init__()
        self.num_heads = num_heads
        _, ch, xL, yL = input_dim
        self.feature_dim = ch * xL * yL // num_heads  # Features per head
        self.linear_q = nn.Linear(ch * xL * yL, ch * xL * yL, bias=False)
        self.linear_k = nn.Linear(ch * xL * yL, ch * xL * yL, bias=False)
        self.linear_v = nn.Linear(ch * xL * yL, ch * xL * yL, bias=False)

    def forward(self, x,interface_map):
        batch_size, ch, xL, yL = x.size()
        q = self.linear_q(x.view(batch_size, -1)).view(batch_size, self.num_heads, self.feature_dim)
        k = self.linear_k(x.view(batch_size, -1)).view(batch_size, self.num_heads, self.feature_dim)
        v = self.linear_v(x.view(batch_size, -1)).view(batch_size, self.num_heads, self.feature_dim)

        # Calculate attention scores
        qk_t = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.feature_dim)
        
        if interface_map is not None:
            interface_map = interface_map.view(batch_size, -1).view(batch_size, self.num_heads, self.feature_dim)
            qk_t *= interface_map  

        attention_weights = torch.softmax(qk_t, dim=-1)
        
        weighted_values = torch.matmul(attention_weights, v)
        combined = weighted_values.transpose(1, 2).contiguous().view(batch_size, ch, xL, yL)
        output = x * combined  # Assuming a simple element-wise multiplication

        return output, attention_weights


"""
class MultiHeadSpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super(MultiHeadSpatialAttention, self).__init__()
        self.input_dim=input_dim
        self.num_heads = 64
        batch_size,ch, xL,yL = input_dim
        self.linear_q = nn.Linear(batch_size*ch* xL*yL, batch_size*ch* xL*yL, bias=False)
        self.linear_k = nn.Linear(batch_size*ch* xL*yL, batch_size*ch* xL*yL, bias=False)
        self.linear_v = nn.Linear(batch_size*ch* xL*yL, batch_size*ch* xL*yL, bias=False)
        
        
    def forward(self, x):
        #print("In attention: ", x.shape)
        batch_size, ch, xL, yL = x.size()
        # get QKV 
        Q = self.linear_q(x.view(batch_size, -1)).view(batch_size, ch, xL, yL)
        K = self.linear_q(x.view(batch_size, -1)).view(batch_size, ch, xL, yL)
        V = self.linear_q(x.view(batch_size, -1)).view(batch_size, ch, xL, yL)
        QKV =  torch.cat((Q, K, V), dim=1)
        #print("QKV: ", QKV.shape)
        QKV = QKV.view(batch_size, -1)
        #print("QKV flattened: ", QKV.shape)
        QKV =QKV.view(batch_size, self.num_heads, -1)
        #print("distributed  QKV: ", QKV.shape)
        head_dim = QKV.size(-1)
        #print("head_dim",head_dim)
        QKV = QKV.view(batch_size, self.num_heads, 3, -1, head_dim // 3)
        #print("1- Reshaped distributed QKV:", QKV.shape)
        QKV = QKV.view(batch_size, self.num_heads, 3, 8, -1)
        #print("2- Reshaped distributed QKV:", QKV.shape)
        #  attention within each head
        d_k = QKV.size(-1)  # Dimension of keys (same as dimension of queries)

        #  q, k, and v tensors
        q = QKV[:, :, 0, :, :]
        k = QKV[:, :, 1, :, :]
        v = QKV[:, :, 2, :, :]
        d_k = q.size(-1)
        #print("d_k",d_k)
        QKt_scaled = torch.matmul(q, k.transpose(-1, -2)) #/ math.sqrt(d_k)
        #print("Shape of QKt_scaled attention :", QKt_scaled.shape)
        
        #print("values:" ,v.shape)
        #  softmax 
        attention_weights = torch.nn.functional.softmax(QKt_scaled, dim=-2)
        attention_scores = torch.nn.functional.softmax(attention_weights, dim=-1)
        #print("Shape of attention weights:", attention_weights.shape)

        # minor check 
        #sum_attention_weights = attention_weights.sum(dim=-1)
        #("Sum of attention weights for each head:", sum_attention_weights)
        
        # 
        weighted_values = attention_scores *v
        #print("Shape of attention_scores:", weighted_values.shape)
        weights_values=weighted_values.view(*self.input_dim)        
        
        attended_representation = x * weights_values
        #("attended_representation  shape:", attended_representation.shape)
                
        return attended_representation
"""
########################################
""" old 
class MultiHeadSpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super(MultiHeadSpatialAttention, self).__init__()
        self.num_heads = 8
        batch_size,ch, xL,yL = input_dim
        self.head_dim = batch_size *ch *xL *yL /self.num_heads 
        out_features =  xL*yL  
        hidden_dim= xL
        in_features = xL 
        self.self_info_net = SelfInformationNetwork(xL  , yL)
        self.relationship = SpatialCorrelation(in_features,out_features)
        self.VT = nn.Linear(out_features, 3 *xL*  self.num_heads, bias=False)  # Linear layer VT
        self.We = nn.Linear(3 *xL*  self.num_heads , 3 *xL*  self.num_heads , bias=False)  # Linear layer We
        self.U = nn.Linear( 3 *xL*  self.num_heads , 3 *xL*  self.num_heads , bias=False)  # Linear layer U
        self.score_linear =nn.Linear( 3*batch_size*ch* xL*yL  , batch_size*ch* xL*yL, bias=False)  # scores
        init.xavier_uniform_(self.VT.weight)
        init.xavier_uniform_(self.We.weight)
        init.xavier_uniform_(self.U.weight)        
        
        
    def forward(self, x):
        print("In attention: ", x.shape)
        batch_size, ch, xL, yL = x.size()
        
        V_vector = self.self_info_net(x)  
        print("V_vector",V_vector.shape)
        QK_components = [x]

        # C´spatial relationships between all pairs of channels using broadcasting
        Qij, Kji = self.relationship(x, x)
        QK_components.extend([Qij, Kji])
        QKV = torch.cat(QK_components, dim=1)
        print("QKV",QKV.shape)
        
        _, ch_, xL, yL = QKV.size()
        QKV = QKV.view(batch_size, ch_ * xL * yL)
        print("QKV",QKV.shape)
        num_heads = self.num_heads
        head_dim = QKV.size(1) // num_heads
        QKV_heads = QKV.view(batch_size, num_heads, head_dim)
        print("QKV_heads",QKV_heads.shape)
        V_vector = V_vector.view(V_vector.size(0), -1) 
        print("VT_out_flattened",V_vector.shape)
        
        # Process each head separately
        head_outputs = []
        for i in range(self.num_heads):
            QKV_head = QKV_heads[:, i, :]
            VT_out = self.VT(V_vector)
            #("---VT_out",VT_out.shape)
            print("---QKV_head",QKV_head.shape)
            We_out = self.We(QKV_head)
            #print("---We_out",We_out.shape)
            U_out = self.U(QKV_head)
            print("---U_out",U_out.shape)
            #scores = torch.matmul(VT_out + We_out + U_out, QKV_head.transpose(-2, -1))
            head_output = torch.tanh(VT_out + We_out + U_out)
            head_outputs.append(head_output)
        
        # Concatenate the outputs of all heads
        QKV = torch.cat(head_outputs, dim=1)
        print("QKV", QKV.shape)
        
        score=self.score_linear(QKV).view(batch_size, ch, xL, yL)
        print("score", score.shape)
        
        
        scaled = score / math.sqrt(ch * xL * yL)
        attention_scores =F.softmax(score, dim=1)
        attention_scores_sum = attention_scores.sum(dim=1)

        #attention_scores=attention_scores.view(batch_size, xL, yL)

        attended_representation = x * attention_scores
        print("attended_representation  shape:", attended_representation.shape)
        stop
        
        return attention_scores
"""

def process_heads(self, QKV_heads, V_vector):
    head_outputs = []
    for i in range(self.num_heads):
        QKV_head = QKV_heads[:, i, :]
        VT_out = self.VT(V_vector)
        We_out = self.We(QKV_head)
        U_out = self.U(QKV_head)
        head_output = torch.tanh(VT_out + We_out + U_out)
        head_outputs.append(head_output)
    return head_outputs
