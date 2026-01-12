import torch
from torch import nn   
import torchvision.transforms as transforms    
from PIL import Image as PILImage
import matplotlib.pyplot as plt            

class Embedding(nn.Module): # Patch + Position Embedding # DONE, paper_relevant_code/vision_transformer.py 
    
    def __init__(
        self, 
        image_size: tuple, 
        patch_size: int, 
        in_channels: int, 
        dim_embed: int, 
        dropout_rate: int = 0,
    ) -> None:
        
        super().__init__()
        H, W = image_size
        num_patches = (H//patch_size) * (W//patch_size)

        self.projection = nn.Conv2d(
            in_channels, dim_embed, 
            kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_embed)) 
        self.position_embedding = nn.Parameter(torch.randn(1, 1+num_patches, dim_embed)) 
        self.dropout = nn.Dropout(dropout_rate) # nécessaire? c'est quoi exactement son utilité?

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.projection(x)          
        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls, x), dim=1)

        x = x + self.position_embedding
        x = self.dropout(x) 

        return x


if __name__ == "__main__":

    patch_size = 32
    num_patches = (224 // patch_size) ** 2  
    emb = Embedding(image_size=(224, 224), patch_size=16, in_channels=3, dim_embed=128)

    # Load and preprocess the image
    img = PILImage.open('louvre.jpg')  # Load the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 224, 224); adding batch dimension
    patched_img = emb.forward(img_tensor)   # shape: (1, 192, 128)
    print("patched_image : ", patched_img.shape)
    first_patch_img = img_tensor[0, :, 120:180, 60:120]

    plt.figure(figsize=(12, 5))

    # Before
    plt.subplot(1, 2, 1)
    plt.imshow(img_tensor[0].permute(1, 2, 0).numpy())
    plt.title('Original Image')
    plt.axis('off')

    # After - show patch grid visualization
    plt.subplot(1, 2, 2)
    plt.imshow(first_patch_img.permute(1, 2, 0).numpy())
    plt.title('Patch Grid Visualization')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
        
    