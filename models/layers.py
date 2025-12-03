
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
        hidden_states = self.act(self.fc1(hidden_states))
        output = self.fc2(hidden_states)
        output = self.dropout(output)
        
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
        batch_size, seq_len, hidden_dim = hidden_states.shape
        qkv_states = self.c_attn(hidden_states) # [batch_size, seq_len, hidden_dim * 3]
        query, key, value = qkv_states.chunk(3, dim=-1) #[batch_size, seq_len, hidden_dim] * 3
        
        def reshape_qkv(data):
            '''
            data [batch_size, seq_len, hidden_dim]
            ret [batch_size * num_head, seq_len, head_dim]
            '''
            data = data.view(batch_size, seq_len, self.num_head, self.head_dim)\
                       .transpose(1, 2)\
                       .reshape(batch_size * self.num_head, seq_len, self.head_dim)
            
            return data
        
        query = reshape_qkv(query) # [batch_size * num_head, seq_len, head_dim]
        key = reshape_qkv(key)     # [batch_size * num_head, seq_len, head_dim]
        value = reshape_qkv(value) # [batch_size * num_head, seq_len, head_dim]
        
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) # [batch_size * num_head, seq_len, seq_len]
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        causal_mask = torch.tril(
                        torch.ones(seq_len, seq_len, device=attn_scores.device, dtype=torch.bool)
                      ).view(1, seq_len, seq_len)   # True: 살릴 곳, False: 가릴 곳
        
        attn_scores = attn_scores.masked_fill(mask=~causal_mask, value=float("-inf"),)
        attn_probs = F.softmax(attn_scores, dim=-1) # [batch_size * num_head, seq_len, seq_len]
        attn_probs = self.dropout(attn_probs)
        
        attn_out = torch.matmul(attn_probs, value) # [batch_size * num_head, seq_len, head_dim]
        
        attn_out = attn_out.reshape(batch_size, self.num_head, seq_len, self.head_dim)\
                           .transpose(1, 2).contiguous()\
                           .reshape(batch_size, seq_len, self.num_head * self.head_dim)
        
        attn_out = self.c_proj(attn_out)
        
        return attn_out
