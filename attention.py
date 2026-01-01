
import math
import pdb
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, attention_head_size, dropout_rate, bias=False):
        super().__init__() # patchedpositioned(x) shape: (batch_size, num_patches+1, embed_dim) = (batch, seq_len, embed_dim)
        self.attention_head_size = attention_head_size
        assert embed_dim % self.attention_head_size == 0
        self.q = nn.Linear(embed_dim, attention_head_size, bias=bias)
        self.v = nn.Linear(embed_dim, attention_head_size, bias=bias)
        self.k = nn.Linear(embed_dim, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,x):
                            # x: (batch, seq_len, embed_dim) @ (embed_dim, attention_head_size) -> (batch, seq_len, attention_head_size)
        query = self.q(x)   # query, ... :(batch, seq_len, attention_head_size)
        value = self.v(x)
        key = self.k(x)
        attention_prob = F.softmax(((query @ key.transpose(-1,-2)) / math.sqrt(self.attention_head_size)), dim=-1)
        attention_prob = self.dropout(attention_prob)
        output = attention_prob @ value # shape : (batch, seq_len, attention_head_size)
        output = self.dropout(output)
        return output, attention_prob
        
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

        attn_heads_cat = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        out = self.out(attn_heads_cat) # shape (B, T, H*d) = (B, T, D) = (batch_size, seq_len, embed_dim)
        out = self.dropout(out)
        attention_probs = torch.stack(  # shape: (batch_size, num_attention_heads, seq_len, seq_len)
            [attention_probs for _, attention_probs in attention_outputs], dim=1)
        return out, attention_probs
        
        

        
        
        
