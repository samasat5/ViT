
import math
import pdb
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from attention import MultiHeadAttention, AugmentedMultiHeadAttention


class MLP(nn.Module):
    def __init__(self, hidden_dim, embed_dim, dropout_rate):
        super().__init__()
        self.denselayer = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.outlayer = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x = self.denselayer(x) # shape: (batch, seq_len, hidden_dim)
        x = self.act(x) 
        x = self.dropout(x)
        x = self.outlayer(x)
        x = self.dropout(x)
        return x



class Norm(nn.Module):
    def __init__(self, embed_dim, gamma_init=1.0,eps=1e-5): 
        super().__init__()
        self.beta = nn.Parameter(torch.ones(embed_dim))
        self.gamma = nn.Parameter(torch.ones(embed_dim) * gamma_init)
        self.eps = eps
    def forward(self, x): 
        mean = x.mean(-1, keepdim=True)  # x:(batch, seq_len, embed_dim);  mean: (batch, seq_len, 1)
        std = x.std(-1, keepdim=True, unbiased=False) # unbiased=False : 1/N sum (xi - mean)^2 ; unbiased=True : 1/(N-1) sum (xi - mean)^2; false is consistent with LayerNorm
        x = (x - mean) / (torch.sqrt(std +  self.eps))
        return self.gamma * x + self.beta



class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_attention_heads, dropout_rate, mlp_hidden_dim, bias=False):
        super().__init__()
        # self.norm1 = nn.LayerNorm(embed_dim)
        self.norm1 = Norm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_attention_heads, dropout_rate, bias=bias)
        # self.norm2 = nn.LayerNorm(embed_dim)
        self.norm2 = Norm(embed_dim)
        self.mlp = MLP(mlp_hidden_dim, embed_dim, dropout_rate)
    def forward(self, x):
        norm1 = self.norm1(x)
        attn_out, attn_probs = self.attn(norm1) # shape: 
        # (batch_size, seq_len, embed_dim), (batch_size, num_attention_heads, seq_len, seq_len)
        x = x + attn_out # shape: (batch_size, seq_len, embed_dim)
        norm2 = self.norm2(x)
        mlp_out = self.mlp(norm2) # shape: (batch_size, seq_len, embed_dim)
        out = x + mlp_out
        return out, attn_probs # (batch_size, seq_len, embed_dim), (batch_size, num_attention_heads, seq_len, seq_len)







class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_attention_heads, dropout_rate, mlp_hidden_dim, bias=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_attention_heads, dropout_rate, mlp_hidden_dim, bias=bias)
            for _ in range(num_layers)])
    def forward(self, x):
        all_attention_probs = []
        for layer in self.layers:
            x, attn_probs = layer(x) 
            all_attention_probs.append(attn_probs) # list of length num_layers, each element shape: (batch_size, num_attention_heads, seq_len, seq_len)
        all_attention_probs = torch.stack(all_attention_probs, dim=1) # shape: (batch_size, num_layers, num_attention_heads, seq_len, seq_len
        return x, all_attention_probs # shape: (batch_size, seq_len, embed_dim), 
                                                # (batch_size, num_layers, num_attention_heads, seq_len, seq_len)





class AugmentedTransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_attention_heads, dropout_rate, mlp_hidden_dim, grid_w, grid_h, bias=False):
        super().__init__()
        # self.norm1 = nn.LayerNorm(embed_dim)
        self.norm1 = Norm(embed_dim)
        self.attn = AugmentedMultiHeadAttention(embed_dim, num_attention_heads, dropout_rate, grid_w, grid_h, bias=bias)
        # self.norm2 = nn.LayerNorm(embed_dim)
        self.norm2 = Norm(embed_dim)
        self.mlp = MLP(mlp_hidden_dim, embed_dim, dropout_rate)
    def forward(self, x):
        norm1 = self.norm1(x)
        attn_out, attn_probs = self.attn(norm1) # shape: 
        # (batch_size, seq_len, embed_dim), (batch_size, num_attention_heads, seq_len, seq_len)
        x = x + attn_out # shape: (batch_size, seq_len, embed_dim)
        norm2 = self.norm2(x)
        mlp_out = self.mlp(norm2) # shape: (batch_size, seq_len, embed_dim)
        out = x + mlp_out
        return out, attn_probs # (batch_size, seq_len, embed_dim), (batch_size, num_attention_heads, seq_len, seq_len)

class AugemntedTransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_attention_heads, dropout_rate, mlp_hidden_dim, grid_w, grid_h, bias=False):
        super().__init__()
        self.layers = nn.ModuleList([
            AugmentedTransformerEncoderBlock(embed_dim, num_attention_heads, dropout_rate, mlp_hidden_dim, grid_w, grid_h, bias=bias)
            for _ in range(num_layers)])
    def forward(self, x):
        all_attention_probs = []
        for layer in self.layers:
            x, attn_probs = layer(x) 
            all_attention_probs.append(attn_probs) # list of length num_layers, each element shape: (batch_size, num_attention_heads, seq_len, seq_len)
        all_attention_probs = torch.stack(all_attention_probs, dim=1) # shape: (batch_size, num_layers, num_attention_heads, seq_len, seq_len
        return x, all_attention_probs # shape: (batch_size, seq_len, embed_dim), 
                                                # (batch_size, num_layers, num_attention_heads, seq_len, seq_len)