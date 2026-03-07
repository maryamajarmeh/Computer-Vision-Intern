import rasterio
import numpy as np
import torch
from PIL import Image


global_min = np.array([-1393, -1169, -722, -684, -412, -335, -251, 64, -9999, 8, 10, 0], dtype=np.float32)
global_max = np.array([6568, 9659, 11368, 12041, 15841, 15252, 14647, 255, 4245, 4287, 100, 111], dtype=np.float32)

def normalize_global(X, gmin, gmax):
    
    X = X.astype(np.float32)
    return (X - gmin) / (gmax - gmin + 1e-8)


def preprocess_image(file):

    with rasterio.open(file) as src:
        img = src.read()  # (12,H,W)
        img = np.transpose(img, (1,2,0))  # (H,W,12)

    
    img = normalize_global(img, global_min, global_max)

    
    img_tensor = torch.tensor(img, dtype=torch.float32)
    img_tensor = img_tensor.permute(2,0,1).unsqueeze(0)  # (1,12,H,W)
    return img_tensor


def mask_to_image(mask_tensor):
   
    mask = mask_tensor.squeeze().cpu().numpy()  # (H,W)
    mask = (mask * 255).astype(np.uint8)       # 0-255
    return Image.fromarray(mask)