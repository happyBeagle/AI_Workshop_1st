
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ACT2FN = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,      # Swish = SiLU
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}

class MLPLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 mlp_dropout_ratio: float,
                 mlp_activation_func: str,
                 ):
        super().__init__()
        act_name = mlp_activation_func.lower()
        mlp_dim = hidden_dim * 4

        if act_name not in ACT2FN:
            raise ValueError(f"Unsupported activation: {act_name}")

        self.fc1 = nn.Linear(in_features=hidden_dim, 
                             out_features=mlp_dim)
        self.act = ACT2FN[act_name]()
        self.fc2 = nn.Linear(in_features=mlp_dim, 
                             out_features=hidden_dim)
        self.dropout = nn.Dropout(p=mlp_dropout_ratio)

    def forward(self, hidden_states):
        #TODO
        return output


class AttentionLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_head: int,
                 attn_dropout_ratio: float,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head
        self.c_attn = nn.Linear(in_features=hidden_dim, 
                                out_features=hidden_dim * 3, 
                                bias=True, 
                                ) 
        self.c_proj = nn.Linear(in_features=hidden_dim, 
                                out_features=hidden_dim, 
                                bias=True, 
                                ) 
        self.dropout = nn.Dropout(p=attn_dropout_ratio)
    
    
    def forward(self, hidden_states):
        '''
        hidden_states [batch_size, seq_len, hidden_dim]
        '''
        #TODO
        
        return attn_out
