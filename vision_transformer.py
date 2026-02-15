import math
import pdb
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from transformer import Transformer
from embed import Embedding


class VisionTransformer(nn.Module): 

    def __init__(
        self, 
        image_size: int, 
        patch_size: int, 
        in_channels: int = 3, 
        dim_embed: int = 768,  
        dim_mlp: int = 512, 
        num_head: int = 12,
        num_transformer: int = 12, 
        num_classes: int = 0, 
        dropout_rate: float = 0., 
        drop_path_rate: float = 0.,
        bias: bool = False, 
        locat: bool = False,
        task: str = 'classif', 
    ) -> None:
 
        super().__init__()
        self.task = task

        self.embedding = Embedding(
            image_size, patch_size, in_channels,
            dim_embed, dropout_rate, task
        )

        grid_size = self.embedding.grid_size
        self.encoder = Transformer(
            grid_size, dim_embed, dim_mlp,
            num_head, num_transformer, dropout_rate,  
            drop_path_rate, bias, locat, task,
        )
        
        if task == "classif":
            self.head = nn.Linear(dim_embed, num_classes)
        else:
            # self.seg_head = nn.Linear(dim_embed, num_classes)  # 1-layer MLP
            self.seg_head = nn.Conv2d(dim_embed, num_classes, kernel_size=1) # à appliquer sur la sortie du transformer (batch_size, seq_len, dim_embed) à reshaper en (batch_size, dim_embed, grid_size, grid_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.embedding(x)  
        x, attn_probs = self.encoder(x) 

        if self.task == "classif":
            cls = x[:, 0]
            return self.head(cls), attn_probs
        
        else: 
            B, N, D = x.shape
            gh, gw = self.embedding.grid_size 
            assert N == gh * gw

            # avec seg_head MLP
            # feat = x  # (B, N, D)
            # logits_tok = self.seg_head(feat) # (B, N, C)
            # logits = logits_tok.transpose(1, 2).contiguous().view(B, self.num_classes, gh, gw)
            # logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

            feat = x.transpose(1, 2).contiguous().view(B, D, gh, gw)

            logits = self.seg_head(feat)
            logits = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )
            return logits, attn_probs


