import math
import pdb
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from attention import MultiHeadAttention


class MLP(nn.Module): # DONE, paper_relevant_code/vision_transformer.py

    def __init__(
        self, 
        dim_embed: int,
        dim_mlp: int, 
        dropout_rate: float,
    ) -> None:
        
        super().__init__()
        self.denselayer = nn.Linear(dim_embed, dim_mlp)
        self.act = nn.GELU()
        self.outlayer = nn.Linear(dim_mlp, dim_embed)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.denselayer(x) # shape: (batch, seq_len, dim_mlp)
        x = self.act(x) 
        x = self.dropout(x)
        x = self.outlayer(x)
        x = self.dropout(x)
        return x


class Norm(nn.Module): # ????

    def __init__(
        self, 
        dim_embed: int, 
        gamma_init: float = 1.0,
        eps: float = 1e-5
    ) -> None: 
        
        super().__init__()
        self.beta = nn.Parameter(torch.ones(dim_embed))
        self.gamma = nn.Parameter(torch.ones(dim_embed) * gamma_init)
        self.eps = eps

    def forward(self, x): 
        mean = x.mean(-1, keepdim=True)  # x:(batch, seq_len, dim_embed);  mean: (batch, seq_len, 1)
        std = x.std(-1, keepdim=True, unbiased=False) # unbiased=False : 1/N sum (xi - mean)^2 ; unbiased=True : 1/(N-1) sum (xi - mean)^2; false is consistent with LayerNorm
        x = (x - mean) / (torch.sqrt(std +  self.eps))
        return self.gamma * x + self.beta


class Encoder(nn.Module): # DONE, Block in paper_relevant_code/vision_transformer.py

    def __init__(
        self, 
        grid_size: int, 
        dim_embed: int, 
        dim_mlp: int, 
        num_head: int, 
        dropout_rate: float, 
        bias: bool = False, 
        locat: bool = True,
        task: str = "classif",
    ) -> None:
        
        super().__init__()
        self.norm1 = Norm(dim_embed)
        self.attn = MultiHeadAttention(
            grid_size, dim_embed, num_head, 
            dropout_rate, bias, locat, task,
        )
        # droppath
        self.norm2 = Norm(dim_embed)
        self.mlp = MLP(dim_embed, dim_mlp, dropout_rate)

    def forward(self, x):
        norm1 = self.norm1(x)
        attn_out, attn_probs = self.attn(norm1) # (batch_size, seq_len, dim_embed), (batch_size, num_head, seq_len, seq_len)
        x = x + attn_out # (batch_size, seq_len, dim_embed)
        norm2 = self.norm2(x)
        mlp_out = self.mlp(norm2) # (batch_size, seq_len, dim_embed)
        out = x + mlp_out
        return out, attn_probs # (batch_size, seq_len, dim_embed), (batch_size, num_head, seq_len, seq_len)


class Transformer(nn.Module): # ajouter PRR
    
    def __init__(
        self,
        grid_size: int, 
        dim_embed: int, 
        dim_mlp: int,
        num_head: int,  
        num_transformer: int, 
        dropout_rate: float,         
        bias: bool = False, 
        locat: bool = False,
        task: str = "classif",
    ) -> None:
        
        super().__init__()
        self.locat = locat
        self.layers = nn.ModuleList([
            Encoder(
                grid_size, dim_embed, dim_mlp, 
                num_head, dropout_rate, bias, locat, task,
            )
            for _ in range(num_transformer)
        ])
        self.norm = nn.LayerNorm(dim_embed)
        # self.prr = 
        
    def forward(self, x):
        all_attention_probs = []
        for layer in self.layers:
            x, attn_probs = layer(x) 
            all_attention_probs.append(attn_probs) # list of length num_transformer, each element shape: (batch_size, num_head, seq_len, seq_len)
        # if not self.locat: all_attention_probs = torch.stack(all_attention_probs, dim=1) # (batch_size, num_transformer, num_head, seq_len, seq_len
        x = self.norm(x)
        return x, all_attention_probs # (batch_size, seq_len, dim_embed), (batch_size, num_transformer, num_head, seq_len, seq_len)