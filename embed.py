
import pdb
import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h, self.w = image_size
        # self.num_patches = (image_size // patch_size) ** 2  #TODO: check if we agree on assuming a square image
        self.num_patches = (self.h // patch_size) * (self.w // patch_size)
        self.projection = nn.Conv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
                                       # x shape (batch_size, in_channels, image_size, image_size)
        x_conv = self.projection(x)            # Shape: (batch_size, out_channels, num_patches_sqrt, num_patches_sqrt)
        x = x_conv.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x                        

class PatchPositionEmbeddings(nn.Module):
    def __init__(self, num_patches, embed_dim, dropout_rate, image_size, patch_size, in_channels):
        super().__init__()
        self.patch_embeddings = PatchEmbedding(
            image_size=image_size,patch_size=patch_size, in_channels=in_channels, out_channels=embed_dim
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable cls token so it can appear in model.parameters()
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches+1, embed_dim)) # Learnable pos emb. so it can appear in model.parameters()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x) # shape: (batch_size, num_patches+1, embed_dim) = (batch, seq_len, embed_dim)
        return x



    
    
    