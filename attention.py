import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from embed import Embedding

# TODO: Fused attention implementation (to speed up training)

class Attention(nn.Module):

    def __init__(
        self, 
        grid_size,
        dim_embed, 
        dim_head, 
        num_prefix_tokens,
        dropout_rate,  
        bias=False, 
        gaussian_augment=False
    ):
        super().__init__() # patchedpositioned(x) shape: (batch_size, num_patches+1, dim_embed) = (batch, seq_len, dim_embed)
        self.gaussian_augment = gaussian_augment
        self.dim_head = dim_head
        assert dim_embed % self.dim_head == 0

        self.q = nn.Linear(dim_embed, dim_head, bias)
        self.v = nn.Linear(dim_embed, dim_head, bias)
        self.k = nn.Linear(dim_embed, dim_head, bias)

        if gaussian_augment:
            self.grid_size = grid_size
            self.num_prefix_tokens = num_prefix_tokens

            self.q_norm = nn.LayerNorm(dim_head)
            self.k_norm = nn.LayerNorm(dim_head)
            self.Wsigma = nn.Linear(dim_head, 2)
            self.Walpha = nn.Linear(dim_head, 1)
        else:
            self.q_norm = nn.LayerNorm(dim_embed)
            self.k_norm = nn.LayerNorm(dim_embed)

        self.dropout = nn.Dropout(dropout_rate) # attention dropout vs projection dropout?

    def forward(self, x): # x: (batch, seq_len, dim_embed) @ (dim_embed, dim_head) -> (batch, seq_len, dim_head)
        query = self.q(x)   # query:(batch, seq_len, dim_head)
        query = self.q_norm(query)
        value = self.v(x)
        key = self.k(x)
        key = self.k_norm(key)
        self.scale = math.sqrt(self.dim_head)

        if self.gaussian_augment:
            query_non_cls = query[:, self.num_prefix_tokens:, :]  
            similarity = (query @ key.transpose(-1,-2)) 
            
            alpha = F.softplus(self.Walpha(query_non_cls))  # shape: (batch, seq_len-cls, 1)
            sigma = torch.sigmoid(self.Wsigma(query_non_cls))  # shape: (batch, seq_len-cls, 2)
            sigma_x = sigma[:, :, 0]  # (batch, seq_len-cls, 1) = (B,N,1)
            sigma_y = sigma[:, :, 1]
            
            W = Embedding.w
            patch_size = Embedding.patch_size
            G = torch.zeros_like(similarity)  # shape: (batch, seq_len, seq_len)
            S = torch.zeros_like(similarity)  # shape: (batch, seq_len, seq_len)
            device = x.device
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
                    

        attention_prob = F.softmax(((query @ key.transpose(-1,-2)) / self.scale), dim=-1)
        attention_prob = self.dropout(attention_prob)
        output = attention_prob @ value # shape : (batch, seq_len, dim_head)
        output = self.dropout(output)

        return output

# https://github.com/pytorch/pytorch/blob/51bdb12f1820879f9f171a726bddbcabe950279f/torch/nn/modules/activation.py#L1091
class MultiHeadAttention(nn.Module):  
    def __init__(
        self, 
        grid_size,
        dim_embed, 
        num_head,
        dropout_rate, 
        device=None,
        bias=False, 
        gaussian_augment=False
    ):
        super().__init__()
        self.dim_head = dim_embed // num_head

        self.heads = nn.ModuleList([
            Attention(
                grid_size, dim_embed, self.dim_head, 
                dropout_rate, bias, gaussian_augment
            )
            for _ in range(num_head)]) 
        self.out = nn.Linear(num_head * self.dim_head, dim_embed) # dim_embed, dim_embed???
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attention_outputs = [head(x) for head in self.heads] 
        # attention_outputs = [
                #     ( (B, T, d), (B, T, T) ),   # head 0; d=dim_head, T=seq_len
                #     ...
                #     ( (B, T, d), (B, T, T) )    # head H-1 ]

        attn_heads_cat = torch.cat([attention_output for attention_output in attention_outputs], dim=-1)
        out = self.out(attn_heads_cat) # shape (B, T, H*d) = (B, T, D) = (batch_size, seq_len, dim_embed)
        out = self.dropout(out)
        # attention_probs = torch.stack(  # shape: (batch_size, num_attention_heads, seq_len, seq_len)
        #     [attention_probs for _, attention_probs in attention_outputs], dim=1)
        return out    
    










        

        
        
        
