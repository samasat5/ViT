import math
import pdb
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from attention import MultiHeadAttention 
from transformer import TransformerEncoder
from embed import PatchPositionEmbeddings

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, 
                 embed_dim, num_attention_heads, mlp_hidden_dim, num_transformer_layers, 
                 dropout_rate, bias=False):
        super().__init__()
        num_patches= (image_size // patch_size) ** 2
        self.embeddings = PatchPositionEmbeddings(num_patches, embed_dim, dropout_rate, 
                                                  image_size, patch_size, in_channels)
        self.encoder = TransformerEncoder(num_transformer_layers, embed_dim, num_attention_heads, 
                                          dropout_rate, mlp_hidden_dim, bias=bias)
        self.linear_head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):       # x shape: (batch_size, in_channels, image_size, image_size)
        x = self.embeddings(x)  # shape: (batch_size, num_patches+1, embed_dim)
        x, attn_probs = self.encoder(x)  #shape: (batch_size, seq_len, embed_dim), 
                                                # (batch_size, num_layers, num_attention_heads, seq_len, seq_len)
        cls_token_final = x[:, 0, :]   # shape: (batch_size, embed_dim)
        logits = self.linear_head(cls_token_final)  # shape: (batch_size, num_classes)
        return logits, attn_probs 
    
    

