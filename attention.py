
import math
import pdb
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from embed import PatchEmbedding

# TODO: Fused attention implementation (to speed up training)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, attention_head_size, dropout_rate, bias=False):
        super().__init__() # patchedpositioned(x) shape: (batch_size, num_patches+1, embed_dim) = (batch, seq_len, embed_dim)
        self.attention_head_size = attention_head_size
        assert embed_dim % self.attention_head_size == 0
        self.q = nn.Linear(embed_dim, attention_head_size, bias=bias)
        self.v = nn.Linear(embed_dim, attention_head_size, bias=bias)
        self.k = nn.Linear(embed_dim, attention_head_size, bias=bias)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,x):
                            # x: (batch, seq_len, embed_dim) @ (embed_dim, attention_head_size) -> (batch, seq_len, attention_head_size)
        query = self.q(x)   # query, ... :(batch, seq_len, attention_head_size)
        query = self.q_norm(query)
        value = self.v(x)
        key = self.k(x)
        key = self.k_norm(key)
        self.scale = math.sqrt(self.attention_head_size)
        attention_prob = F.softmax(((query @ key.transpose(-1,-2)) / self.scale), dim=-1)
        attention_prob = self.dropout(attention_prob)
        output = attention_prob @ value # shape : (batch, seq_len, attention_head_size)
        output = self.dropout(output)
        return output


class MultiHeadAttention(nn.Module):  
    def __init__(self, embed_dim, num_attention_heads, dropout_rate, bias=False):
        super().__init__()
        self.attention_head_size = embed_dim // num_attention_heads
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, self.attention_head_size, dropout_rate, bias=bias)
            for _ in range(num_attention_heads)]) 
        self.out = nn.Linear(num_attention_heads * self.attention_head_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        attention_outputs = [head(x) for head in self.heads] 
        # attention_outputs = [
                #     ( (B, T, d), (B, T, T) ),   # head 0; d=attention_head_size, T=seq_len
                #     ...
                #     ( (B, T, d), (B, T, T) )    # head H-1 ]

        attn_heads_cat = torch.cat([attention_output for attention_output in attention_outputs], dim=-1)
        out = self.out(attn_heads_cat) # shape (B, T, H*d) = (B, T, D) = (batch_size, seq_len, embed_dim)
        out = self.dropout(out)
        # attention_probs = torch.stack(  # shape: (batch_size, num_attention_heads, seq_len, seq_len)
        #     [attention_probs for _, attention_probs in attention_outputs], dim=1)
        return out 
    
    
class AugmentedAttentionHead(nn.Module):
    def __init__(self, embed_dim, attention_head_size, dropout_rate, grid_w, grid_h,num_prefix_tokens=1, bias=False):
        super().__init__() # patchedpositioned(x) shape: (batch_size, num_patches+1, embed_dim) = (batch, seq_len, embed_dim)
        self.attention_head_size = attention_head_size
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.num_prefix_tokens = num_prefix_tokens
        
        self.q = nn.Linear(embed_dim, attention_head_size, bias=bias)
        self.v = nn.Linear(embed_dim, attention_head_size, bias=bias)
        self.k = nn.Linear(embed_dim, attention_head_size, bias=bias)
        self.q_norm = nn.LayerNorm(attention_head_size)
        self.k_norm = nn.LayerNorm(attention_head_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.Wsigma = nn.Linear(attention_head_size, 2)
        self.Walpha = nn.Linear(attention_head_size, 1)
        
    def forward(self,x):
        
                            # x: (batch, seq_len, embed_dim) @ (embed_dim, attention_head_size) -> (batch, seq_len, attention_head_size)
        query = self.q(x)   # query, ... :(batch, seq_len, attention_head_size)
        query = self.q_norm(query)
        value = self.v(x)
        key = self.k(x)
        key = self.k_norm(key)
        self.scale = math.sqrt(self.attention_head_size)
        query_non_cls = query[:, self.num_prefix_tokens:, :]  

        
        similarity = (query @ key.transpose(-1,-2)) 
        
        
        alpha = F.softplus(self.Walpha(query_non_cls))  # shape: (batch, seq_len-cls, 1)
        sigma = torch.sigmoid(self.Wsigma(query_non_cls))  # shape: (batch, seq_len-cls, 2)
        sigma_x = sigma[:, :, 0]  # (batch, seq_len-cls, 1) = (B,N,1)
        sigma_y = sigma[:, :, 1]
        
        W = PatchEmbedding.w
        patch_size = PatchEmbedding.patch_size
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
                
                
        attention_prob = F.softmax((similarity / self.scale), dim=-1)  # shape: (batch, seq_len, seq_len)
        attention_prob = self.dropout(attention_prob)
        output = attention_prob @ value # shape : (batch, seq_len, attention_head_size)
        output = self.dropout(output)
        return output









        

        
        
        
