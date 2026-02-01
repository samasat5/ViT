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


class VisionTransformer(nn.Module): # paper_relevant_code/vision_transformer.py

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
        dropout_rate: int = 0, 
        bias: bool = False, 
        locat: bool = False,
        task: str = 'classif', # classif or seg
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
            bias, locat, task,
        )
        
        self.head = nn.Linear(dim_embed, num_classes)

    def forward(self, x): # x: (batch_size, in_channels, image_size, image_size)
        x = self.embedding(x)  # (batch_size, 1+num_patches, dim_embed)
        x, attn_probs = self.encoder(x) #(batch_size, seq_len, dim_embed), (batch_size, num_layers, num_head, seq_len, seq_len)

        if self.task == "classif":
            cls = x[:, 0]
            return self.head(cls), attn_probs
        else:
            return x, attn_probs   # puis un head de seg ailleurs


