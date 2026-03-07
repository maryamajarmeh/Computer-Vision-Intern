import torch
import segmentation_models_pytorch as smp


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PATH = "unet_efficientnetb0_12bands.pth"

def get_model():
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,   
        in_channels=12,
        classes=1
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  
    return model

def predict_mask(model, image_tensor, threshold=0.5):
    """
    image_tensor: torch.Tensor shape (1,12,H,W) or (12,H,W)
    threshold: 0-1 to specify binary mask
    """
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # (1,12,H,W)

    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(image_tensor))  # (1,1,H,W)
        pred_mask = (pred > threshold).float()    # binary mask
    return pred_mask.cpu() 