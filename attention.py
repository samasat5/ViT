import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module): 

    def __init__(
        self, 
        grid_size: tuple,
        dim_embed: int, 
        num_head: int, 
        dropout_rate: float = 0.,  
        bias: bool = False, 
        locat: bool = False,
        task: str = "classif",
    ) -> None:
        
        super().__init__()
        self.locat = locat
        self.num_head = num_head
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_head
        self.task = task
        self.num_prefix_tokens = 1 if task == "classif" else 0
        assert dim_embed % num_head == 0

        self.q = nn.Linear(dim_embed, dim_embed, bias=bias)
        self.v = nn.Linear(dim_embed, dim_embed, bias=bias)
        self.k = nn.Linear(dim_embed, dim_embed, bias=bias)

        if locat:
            self.w_var = nn.Linear(self.dim_head, 2)
            self.w_alpha = nn.Linear(self.dim_head, 1)

            self.x_grid, self.y_grid = grid_size
            pixels = torch.stack(
                torch.meshgrid(
                    torch.arange(self.x_grid),
                    torch.arange(self.y_grid),
                    indexing="ij",
                ),
                dim=-1)
            diff = pixels.unsqueeze(0).unsqueeze(1) - pixels.unsqueeze(2).unsqueeze(3)
            self.register_buffer("diff", diff.pow(2), persistent=False)  

        self.dropout = nn.Dropout(dropout_rate) 

    def forward(self, x): 
        B, N, E = x.shape 
        assert E == self.dim_embed

        value = self.v(x).reshape(B, N, self.num_head, self.dim_head).transpose(1, 2)
        key = self.k(x).reshape(B, N, self.num_head, self.dim_head).transpose(1, 2)
        query = self.q(x).reshape(B, N, self.num_head, self.dim_head).transpose(1, 2) 

        addition = None
        if self.locat:
            eps = 1e-6
            q_loc = query[:, :, self.num_prefix_tokens:, :] 
            var = F.softplus(self.w_var(q_loc)) + eps 
            var = var.unsqueeze(3)
            alpha = F.softplus(self.w_alpha(q_loc)) 

            diff = self.diff.to(dtype=x.dtype)
            distances = -1/2 * diff 
            N = self.x_grid * self.y_grid
            distances = distances.reshape(1, 1, N, N, 2) 
            
            B, H, N, _ = alpha.shape
            gaussian = torch.exp((distances / (var + eps)).sum(dim=-1)) 
            addition = alpha * gaussian 
            addition = addition.to(dtype=query.dtype, device=query.device)
            assert addition.shape == (B, H, N, N)

            if self.num_prefix_tokens > 0: 
                addition = F.pad(
                    addition, 
                    pad=(self.num_prefix_tokens, 0, self.num_prefix_tokens, 0), 
                    mode='constant',
                ) 

        x = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout.p if self.training else 0.,
            attn_mask=addition,
        )

        attention_prob = None
        x = x.transpose(1, 2).flatten(start_dim=2)
        return x, attention_prob


class MultiHeadAttention(nn.Module):  
    def __init__(
        self, 
        grid_size: int,
        dim_embed: int, 
        num_head: int,
        dropout_rate: float, 
        bias: bool = False, 
        locat: bool = False,
        task: str = "classif",
    ) -> None:
        
        super().__init__()

        self.heads = Attention(
            grid_size, dim_embed, num_head, 
            dropout_rate, bias, locat, task,
        ) 
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(dim_embed, dim_embed, bias=bias) 

    def forward(self, x):
        x, attn_prob = self.heads(x)
        x = self.out(x) 
        x = self.dropout(x)
        return x, attn_prob    