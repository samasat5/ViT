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


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    
    # masque par sample
    batch_size = x.shape[0]
    mask = torch.rand(batch_size, 1, 1, device=x.device) < keep_prob
    mask = mask.to(x.dtype)

    return x * mask / keep_prob

class Encoder(nn.Module): # DONE, Block in paper_relevant_code/vision_transformer.py

    def __init__(
        self, 
        grid_size: int, 
        dim_embed: int, 
        dim_mlp: int, 
        num_head: int, 
        dropout_rate: float, 
        drop_path_rate: float = 0.,
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
        self.drop_path_rate = drop_path_rate
        self.norm2 = Norm(dim_embed)
        self.mlp = MLP(dim_embed, dim_mlp, dropout_rate)

    def forward(self, x):
        norm1 = self.norm1(x)
        attn_out, attn_probs = self.attn(norm1) # (batch_size, seq_len, dim_embed), (batch_size, num_head, seq_len, seq_len)
        
        # DropPath sur la branche d'attention
        x = x + drop_path(attn_out, self.drop_path_rate, self.training) # (batch_size, seq_len, dim_embed)

        norm2 = self.norm2(x)
        mlp_out = self.mlp(norm2) # (batch_size, seq_len, dim_embed)
        
        # DropPath sur la branche MLP
        out = x + drop_path(mlp_out, self.drop_path_rate, self.training) # (batch_size, seq_len, dim_embed)

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
        drop_path_rate: float = 0.,      
        bias: bool = False, 
        locat: bool = False,
        task: str = "classif",
    ) -> None:
        
        super().__init__()
        self.locat = locat
        drop_path_encoder = torch.linspace(0, drop_path_rate, num_transformer).tolist()
        self.layers = nn.ModuleList([
            Encoder(
                grid_size, dim_embed, dim_mlp, 
                num_head, dropout_rate, drop_path_encoder[i], bias, locat, task,
            )
            for i in range(num_transformer)
        ])
        self.norm = nn.LayerNorm(dim_embed)

        # PRR : bloc de raffinnement positionnel
        self.num_head = num_head
        self.pre_norm, self.post_norm = nn.Identity(), nn.Identity()
        # nn.Identity in PyTorch is a layer that simply returns the input as output 
        # without any changes. It is often used in neural networks to create skip 
        # connections or as a placeholder in model architectures.
        
    def forward(self, x):
        all_attention_probs = []
        for layer in self.layers:
            x, attn_probs = layer(x) 
            all_attention_probs.append(attn_probs) # list of length num_transformer, each element shape: (batch_size, num_head, seq_len, seq_len)
        # if not self.locat: all_attention_probs = torch.stack(all_attention_probs, dim=1) # (batch_size, num_transformer, num_head, seq_len, seq_len
        x = self.norm(x)
        # PRR
        if self.locat: 
            x = self.pre_norm(x)
            B, N, C = x.shape
            x = x.view(B, N, self.num_head, -1).permute(0, 2, 1, 3)
            x = F.scaled_dot_product_attention(
                query=x, key=x, value=x,
            )
            x = x.permute(0, 2, 1, 3).flatten(2, 3)
            x = self.post_norm(x)
            x = x.reshape((B, N, C))

        return x, all_attention_probs # (batch_size, seq_len, dim_embed), (batch_size, num_transformer, num_head, seq_len, seq_len)