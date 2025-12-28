
import pdb
from PIL import Image as PILImage
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        print(x.shape)
                                       # x shape (batch_size, in_channels, image_size, image_size)
        x_conv = self.projection(x)            # Shape: (batch_size, out_channels, num_patches_sqrt, num_patches_sqrt)
        x = x_conv.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, out_channels)
        print(x.shape)
        print(self.projection.weight.shape)
        return x                        


patch_size=32
num_patches = (224 // patch_size) ** 2  # 
emb = PatchEmbedding(image_size=224, patch_size=32, in_channels=3, out_channels=768)

# Load and preprocess the image
img = PILImage.open('louvre.jpg')  # Load the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img_tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 224, 224); adding batch dimension
patched_img = emb.forward(img_tensor)   # shape: (1, 784, 768)
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