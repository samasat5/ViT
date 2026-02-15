import torch
from torch import nn   
import torchvision.transforms as transforms    
from PIL import Image as PILImage
import matplotlib.pyplot as plt            

class Embedding(nn.Module): 
    
    def __init__(
        self, 
        image_size: tuple, 
        patch_size: tuple, 
        in_channels: int, 
        dim_embed: int, 
        dropout_rate: int = 0,
        task: str = "classif", 
    ) -> None:
        
        super().__init__()
        H, W = image_size
        P_H, P_W = patch_size
        N = (H//P_H) * (W//P_W)

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (H//P_H, W//P_W)
        self.task = task

        self.projection = nn.Conv2d(
            in_channels, dim_embed, 
            kernel_size=patch_size, stride=patch_size
        )

        if task == "classif":
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim_embed))
            self.position_embedding = nn.Parameter(
                torch.randn(1, 1 + N, dim_embed)
            )
        else: 
            self.cls_token = None
            self.position_embedding = nn.Parameter(
                torch.randn(1, N, dim_embed)
            )

        self.dropout = nn.Dropout(dropout_rate) 

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.projection(x)          
        x = x.flatten(2).transpose(1, 2)

        if self.task == "classif":
            cls = self.cls_token.expand(B, -1, -1) 
            x = torch.cat((cls, x), dim=1)

        assert (H, W) == self.image_size

        x = x + self.position_embedding
        x = self.dropout(x) 

        return x
