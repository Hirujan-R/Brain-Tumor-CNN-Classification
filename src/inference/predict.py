import torch
import torch.nn as nn
import numpy as np
from src.models.googlenet import GoogLeNetBrainTumor

def load_model(model_path="model/model.pth", num_classes=3, device="cpu"):
    # Instantiate the exact same class used at training time
    model = GoogLeNetBrainTumor(num_classes=num_classes, pretrained=False, aux_logits=True)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # The .pth has "model." prefix because GoogLeNetBrainTumor uses self.model internally.
    # Load directly — no key manipulation needed.
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()
    return model


def predict(model, image_tensor, device="cpu"):
    model.eval()

    if not isinstance(image_tensor, np.ndarray):
        raise TypeError("image_tensor must be a numpy array.")

    if image_tensor.ndim != 3 or image_tensor.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got {image_tensor.shape}")

    tensor = torch.tensor(image_tensor, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        outputs = model(tensor)
        # GoogLeNetBrainTumor may return a tuple/named tuple — always unwrap
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return pred_class, probs.squeeze().cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(
        "model/model.pth",
        num_classes=3,
        device=device
    )
    image_tensor = np.load("data/processed/images/000001.npy")

    pred, probs = predict(model, image_tensor=image_tensor, device=device)
    print(f"Predicted class: {pred}")
    print(f"Probabilities: {probs}")  


if __name__ == "__main__":
    main()