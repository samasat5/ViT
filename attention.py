import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from embed import Embedding

# TODO: Device, gaussian, 

class Attention(nn.Module):

    def __init__(
        self, 
        grid_size: int,
        dim_embed: int, 
        num_head: int, 
        num_prefix_tokens: int,
        dropout_rate: float,  
        bias: bool = False, 
        locat: bool = False,
    ) -> None:
        
        super().__init__() # patchedpositioned(x) shape: (batch_size, num_patches+1, dim_embed) = (batch, seq_len, dim_embed)
        self.locat = locat
        self.num_head = num_head
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_head
        self.scale = math.sqrt(self.dim_head)
        assert dim_embed % self.dim_head == 0

        self.q = nn.Linear(dim_embed, dim_embed, bias=bias)
        self.v = nn.Linear(dim_embed, dim_embed, bias=bias)
        self.k = nn.Linear(dim_embed, dim_embed, bias=bias)

        if locat:
            self.grid_size = grid_size
            self.num_prefix_tokens = num_prefix_tokens

            self.w_var = nn.Linear(self.dim_head, 2)
            self.w_alpha = nn.Linear(self.dim_head, 1)

        self.dropout = nn.Dropout(dropout_rate) # attention dropout vs projection dropout?

    def forward(self, x): 
        B, N, E = x.shape # (B, N, E) => la forme canonique multihead : (B, H, N, Dh)
        assert E == self.dim_embed

        # Le reshape sert à découper E en H morceaux de taille Dh(dim_head)
        value = self.v(x).reshape(B, N, self.num_head, self.dim_head).transpose(1, 2)
        key = self.k(x).reshape(B, N, self.num_head, self.dim_head).transpose(1, 2)
        query = self.q(x).reshape(B, N, self.num_head, self.dim_head).transpose(1, 2)

        if self.locat:
            query_non_cls = query[:, self.num_prefix_tokens:, :]  
            similarity = (query @ key.transpose(-1,-2)) 
            
            alpha = F.softplus(self.w_alpha(query_non_cls))  # shape: (batch, seq_len-cls, 1)
            sigma = torch.sigmoid(self.w_var (query_non_cls))  # shape: (batch, seq_len-cls, 2)
            sigma_x = sigma[:, :, 0]  # (batch, seq_len-cls, 1) = (B,N,1)
            sigma_y = sigma[:, :, 1]
            
            W = Embedding.w
            patch_size = Embedding.patch_size
            G = torch.zeros_like(similarity)  # shape: (batch, seq_len, seq_len)
            S = torch.zeros_like(similarity)  # shape: (batch, seq_len, seq_len)
            self.device = x.device
            coords = torch.stack(torch.meshgrid(
            torch.arange(self.grid_h),
            torch.arange(self.grid_w),
            indexing="ij"), dim=-1).reshape(self.grid_h * self.grid_w, 2) 
            
            for p in range(x.shape[1] - self.num_prefix_tokens): # seq_len-cls
                for q in range(x.shape[1] - self.num_prefix_tokens):
                    patch_p_ind = p
                    patch_q_ind = q
                    diff_pq_x = (coords[patch_p_ind] - coords[patch_q_ind])[1].float()
                    diff_pq_y = (coords[patch_p_ind] - coords[patch_q_ind])[0].float()
                    G[:, p, q] = torch.exp(- (diff_pq_x ** 2) / (
                        2 * sigma_x[:, patch_p_ind] ** 2) - (diff_pq_y ** 2) / (2 * sigma_y[:, patch_p_ind] ** 2))  # shape: (batch, seq_len-cls, 1)
                    S[:, p, q] = alpha[:, p, 0] * G[:, p, q]  # shape: (batch, seq_len-cls, 1)
                    similarity[:, p+self.num_prefix_tokens, q+self.num_prefix_tokens] += S[:, p, q]
                    


        # (B, H, N, Dh)
        x = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout.p if self.training else 0.,
        )
        attention_prob = None

        # attention_prob = F.softmax(((query @ key.transpose(-1, -2)) / self.scale), dim=-1)
        # attention_prob = self.dropout(attention_prob)
        # output = attention_prob @ value # shape : (batch, seq_len, dim_head)
        # output = self.dropout(output)

        x = x.transpose(1, 2).flatten(start_dim=2)

        return x, attention_prob

# https://github.com/pytorch/pytorch/blob/51bdb12f1820879f9f171a726bddbcabe950279f/torch/nn/modules/activation.py#L1091
class MultiHeadAttention(nn.Module):  
    def __init__(
        self, 
        grid_size: int,
        dim_embed: int, 
        num_head: int,
        dropout_rate: float, 
        bias: bool = False, 
        locat: bool = False,
    ) -> None:
        
        super().__init__()
        self.dim_head = dim_embed // num_head

        self.heads = Attention(
            grid_size, dim_embed, self.dim_head, 
            dropout_rate, bias, locat
        ) 
        self.out = nn.Linear(num_head * self.dim_head, dim_embed) # dim_embed, dim_embed???
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x, _ = self.heads(x)
        x = self.out(x) 
        x = self.dropout(x)
        return x         