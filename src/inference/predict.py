import torch
from torchvision.models import googlenet
import numpy as np


def load_model(model_path="model/model.pth", num_classes=3, device="cpu"):
    model = googlenet(weights=None)

    # IMPORTANT: match training architecture exactly
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.aux1.fc2 = torch.nn.Linear(model.aux1.fc2.in_features, num_classes)
    model.aux2.fc2 = torch.nn.Linear(model.aux2.fc2.in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)

    # remove "model." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("model.", "")] = v

    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()

    return model



def predict(model, image_tensor, device="cpu", preprocessed=True):
    """
    Make prediction on an image.
    
    Args:
        model: Loaded PyTorch model
        image_tensor: Input image as numpy array
        device: Device to use for inference
        preprocessed: If True, assumes image is already preprocessed (from .npy files).
                     If False, will apply raw preprocessing pipeline.
    
    Returns:
        pred_class: Predicted class (0, 1, or 2)
        probs: Class probabilities as numpy array
    """
    model.eval()

    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
    else:
        raise TypeError("image_tensor must be a np array.")

    # Convert HWC -> CHW format for PyTorch
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

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
    pred_class, probs = predict(
        model,
        image_tensor= image_tensor,
        device=device
    )

    print(pred_class)
    print(probs)

if __name__ == "__main__":
    main()