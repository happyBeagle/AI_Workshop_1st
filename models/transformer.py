import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AttentionLayer, MLPLayer

class TransformerBlock(nn.Module):
    def __init__(self,
                 layer_idx: int,
                 hidden_dim: int,
                 num_head: int,
                 attn_dropout_ratio: float,
                 mlp_dropout_ratio: float,
                 mlp_activation_func: str,
                 layer_norm_epsilon: float=1e-5,
                 ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(normalized_shape=hidden_dim, 
                                 eps=layer_norm_epsilon, 
                             ) 
        self.attn = AttentionLayer(hidden_dim=hidden_dim,
                                   num_head=num_head,
                                   attn_dropout_ratio=attn_dropout_ratio,
                             )
        self.ln_2 = nn.LayerNorm(normalized_shape=hidden_dim, 
                                 eps=layer_norm_epsilon, 
                             ) 
        self.mlp = MLPLayer(hidden_dim=hidden_dim,
                            mlp_dropout_ratio=mlp_dropout_ratio,
                            mlp_activation_func=mlp_activation_func
                             )
        
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        attn_output = self.attn(hidden_states)
        
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        
        hidden_states = feed_forward_hidden_states + residual
        
        return hidden_states
        
class TransformerModel(nn.Module): 
    def __init__(self, 
                 num_layers: int, 
                 hidden_dim: int,
                 num_head: int, 
                 emb_dropout_ratio: float, 
                 attn_dropout_ratio: float, 
                 mlp_dropout_ratio: float,
                 max_seq_len: int, 
                 vocab_size: int,
                 mlp_activation_func: str,
                 layer_norm_epsilon: float = 1e-5, 
                 ): 
        super().__init__() 
        self.num_layers = num_layers 
        self.wte = nn.Embedding(num_embeddings=vocab_size, 
                                embedding_dim=hidden_dim, 
                                ) 
        self.wpe = nn.Embedding(num_embeddings=max_seq_len, 
                                embedding_dim=hidden_dim, 
                                ) 
        self.drop = nn.Dropout(p=emb_dropout_ratio) 
        self.blocks = nn.ModuleList([ 
                        TransformerBlock(layer_idx=i, 
                                        hidden_dim=hidden_dim, 
                                        num_head=num_head, 
                                        attn_dropout_ratio=attn_dropout_ratio, 
                                        mlp_dropout_ratio=mlp_dropout_ratio, 
                                        mlp_activation_func=mlp_activation_func, 
                                        layer_norm_epsilon=layer_norm_epsilon,
                                        ) for i in range(num_layers)]) 
        self.ln_f = nn.LayerNorm(normalized_shape=hidden_dim, 
                                eps=layer_norm_epsilon, 
                                )
        
    def forward(self, token_ids): 
        '''
        token_ids [batch_size, seq_len] 
        ''' 
        batch_size, seq_len = token_ids.shape 
        position_ids = torch.arange(seq_len, device=token_ids.device, dtype=torch.long).unsqueeze(0) 
        position_ids = position_ids.expand(batch_size, seq_len) 
        
        position_emb = self.wpe(position_ids) 
        token_emb = self.wte(token_ids) 
        
        hidden_states = position_emb + token_emb.to(position_emb.device) 
        
        hidden_states = self.drop(hidden_states) 
        for block in self.blocks: 
            hidden_states = block(hidden_states) 
        
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states 


class Transformer(nn.Module): 
    def __init__(self, 
                 num_layers: int,
                 hidden_dim: int,
                 num_head: int, 
                 emb_dropout_ratio: float, 
                 attn_dropout_ratio: float, 
                 mlp_dropout_ratio: float, 
                 vocab_size: int, 
                 max_seq_len: int, 
                 mlp_activation_func: str, 
                 layer_norm_epsilon: float = 1e-5,
                 ): 
        super().__init__() 
        self.transformer = TransformerModel(num_layers=num_layers, 
                                            hidden_dim=hidden_dim, 
                                            num_head=num_head, 
                                            emb_dropout_ratio=emb_dropout_ratio, 
                                            attn_dropout_ratio=attn_dropout_ratio, 
                                            mlp_dropout_ratio=mlp_dropout_ratio, 
                                            vocab_size=vocab_size, 
                                            max_seq_len=max_seq_len, 
                                            mlp_activation_func=mlp_activation_func,
                                            layer_norm_epsilon=layer_norm_epsilon,
                                            ) 
        self.lm_head = nn.Linear(in_features=hidden_dim, 
                                out_features=vocab_size, 
                                bias=False, 
                                ) 
    
    def forward(self, token_ids): 
        hidden_state = self.transformer(token_ids) 
        logits = self.lm_head(hidden_state) 
        return logits