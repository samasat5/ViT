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


class VisionTransformer(nn.Module): # forward????, paper_relevant_code/vision_transformer.py

    def __init__(
        self, 
        image_size, 
        patch_size, 
        grid_size, 
        in_channels, 
        dim_embed,  
        dim_mlp, 
        num_head,
        num_transformer, 
        num_classes, 
        dropout_rate, 
        bias=False, 
        gaussian_augment=False
    ):
        super().__init__()

        self.embedding = Embedding(
            image_size, patch_size, in_channels,
            dim_embed, dropout_rate, 
        )
        self.encoder = Transformer(
            grid_size, dim_embed, dim_mlp,
            num_head, num_transformer, dropout_rate,  
            bias, gaussian_augment
        )
        self.head = nn.Linear(dim_embed, num_classes)

    def forward(self, x): # x: (batch_size, in_channels, image_size, image_size)
        # TODO: grid size propagation, revoir prepare_tokens
        x = self.embedding(x)  # (batch_size, 1+num_patches, dim_embed)
        x, attn_probs = self.encoder(x) #(batch_size, seq_len, dim_embed), (batch_size, num_layers, num_head, seq_len, seq_len)
        x = self.norm(x)
        cls_token_final = x[:, 0, :] # (batch_size, dim_embed)
        logits = self.head(cls_token_final) # (batch_size, num_classes)

        return logits, attn_probs 

