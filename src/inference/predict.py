import os
import torch
from torchvision.models import googlenet
import numpy as np
from typing import Union, Dict, Any, Tuple
from pathlib import Path

# Project-level imports
from src.datasets.label_mapping import decode_label, MODEL_LABEL_TO_ORIGINAL_LABEL
from src.preprocessing.image_loader import load_mat_image
from src.preprocessing.transforms import (
    ensure_grayscale,
    zscore_normalize,
    minmax_normalize,
    crop_to_brain,
    resize_image,
    replicate_channels
)
from src.preprocessing.config import NORMALIZATION

def preprocess_raw_image(image: np.ndarray, clip_compatibility: bool = True) -> np.ndarray:
    """
    Applies the full preprocessing pipeline on a raw numpy array to make it
    compatible with the model inputs.

    Args:
        image (np.ndarray): Raw input image array.
        clip_compatibility (bool): If True, clips the z-scored values to [0, 1] 
            to match the training data of the current model.pth.

    Returns:
        np.ndarray: Preprocessed 3-channel image array of shape (224, 224, 3).
    """
    # 1. Convert to grayscale
    image = ensure_grayscale(image)
    # 2. Normalize using the same strategy as preprocessing pipeline
    if NORMALIZATION == "zscore":
        image = zscore_normalize(image)
    elif NORMALIZATION == "minmax":
        image = minmax_normalize(image)
    else:
        # fallback to z-score to avoid silent incorrect scaling
        image = zscore_normalize(image)
    # 3. Crop to brain area
    image = crop_to_brain(image)
    # 4. Resize to 224x224
    image = resize_image(image, target_height=224, target_width=224, interpolation="bilinear", clip_to_0_1=clip_compatibility)
    # 5. Replicate to 3 channels
    image = replicate_channels(image, channels=3)
    
    return image

def load_model(model_path="model/model.pth", num_classes=3, device="cpu"):
    """
    Loads the trained GoogLeNet model weights into the model architecture.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    model = googlenet(weights=None)

    # Match training architecture exactly
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.aux1.fc2 = torch.nn.Linear(model.aux1.fc2.in_features, num_classes)
    model.aux2.fc2 = torch.nn.Linear(model.aux2.fc2.in_features, num_classes)

    raw = torch.load(model_path, map_location=device)

    # Support both plain state_dicts and checkpoint dicts that contain nested keys
    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw
    else:
        state_dict = raw

    # Normalize common prefixes from typical training wrappers (e.g., 'model.', 'module.')
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model

def predict(
    model: torch.nn.Module,
    image_input: Union[np.ndarray, torch.Tensor, str, Path],
    device: str = "cpu",
    preprocess_raw: bool = False,
    clip_compatibility: bool = True
) -> Dict[str, Any]:
    """
    Runs inference on the provided image input (numpy array, tensor, or file path).

    Args:
        model (torch.nn.Module): Loaded model.
        image_input (Union[np.ndarray, torch.Tensor, str, Path]): Image array or path to (.npy or .mat) file.
        device (str): Device to run inference on ('cpu', 'cuda', etc.).
        preprocess_raw (bool): If True, runs the preprocessing pipeline on the input.
            Automatically set to True for .mat files.
        clip_compatibility (bool): Whether to clip z-score values to [0,1] during raw preprocessing.

    Returns:
        Dict[str, Any]: Dictionary containing prediction results:
            - "predicted_label" (int): Zero-based model prediction (0, 1, or 2).
            - "original_label" (int): Original Figshare prediction space (1, 2, or 3).
            - "class_name" (str): Decoded class name ("glioma", "meningioma", "pituitary").
            - "probabilities" (np.ndarray): Softmax probabilities.
    """
    model.eval()
    
    # 1. Load image if path is provided
    if isinstance(image_input, (str, Path)):
        image_path = str(image_input)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")
            
        if image_path.endswith(".mat"):
            image_input = load_mat_image(image_path)
            preprocess_raw = True  # Always preprocess raw .mat files
        elif image_path.endswith(".npy"):
            image_input = np.load(image_path)
        else:
            raise ValueError("Unsupported file format. Must be .mat or .npy")

    # 2. Preprocess raw numpy array if requested
    if preprocess_raw:
        if not isinstance(image_input, np.ndarray):
            raise TypeError("image_input must be a numpy array to preprocess.")
        image_input = preprocess_raw_image(image_input, clip_compatibility=clip_compatibility)

    # 3. Convert to torch Tensor if it is a numpy array
    if isinstance(image_input, np.ndarray):
        image_tensor = torch.tensor(image_input, dtype=torch.float32)
    elif isinstance(image_input, torch.Tensor):
        image_tensor = image_input.to(dtype=torch.float32)
    else:
        raise TypeError("image_input must be a numpy array, torch Tensor, or file path.")

    # 4. Handle Shape and Permutation
    # Expected shape for model is (C, H, W).
    # If the input has shape (H, W, C), permute to (C, H, W).
    # Typically, a 3D tensor of shape (224, 224, 3) is (H, W, C), while (3, 224, 224) is (C, H, W).
    if image_tensor.ndim == 3:
        if image_tensor.shape[2] in (1, 3) and image_tensor.shape[0] not in (1, 3):
            # Shape is (H, W, C)
            image_tensor = image_tensor.permute(2, 0, 1)
        elif image_tensor.shape[0] in (1, 3):
            # Shape is already (C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")
        # Add batch dimension: (1, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.ndim == 4:
        # Already has batch dimension: (B, C, H, W) or (B, H, W, C)
        if image_tensor.shape[3] in (1, 3) and image_tensor.shape[1] not in (1, 3):
            image_tensor = image_tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Image tensor must be 3D or 4D, got {image_tensor.ndim}D shape: {image_tensor.shape}")

    # Send to device
    image_tensor = image_tensor.to(device)

    # 5. Model Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        pred_class = int(torch.argmax(outputs, dim=1).item())

    # 6. Decode output labels
    class_name = decode_label(pred_class)
    original_label = MODEL_LABEL_TO_ORIGINAL_LABEL[pred_class]

    return {
        "predicted_label": pred_class,
        "original_label": original_label,
        "class_name": class_name,
        "probabilities": probs
    }

def main():
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Predict brain tumor class from an MRI scan image.")
    parser.add_argument("--image", type=str, default="data/processed/images/000001.npy",
                        help="Path to .mat or .npy image file to run prediction on.")
    parser.add_argument("--model", type=str, default="model/model.pth",
                        help="Path to trained model checkpoint file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use ('cpu' or 'cuda').")
    parser.add_argument("--preprocess", action="store_true",
                        help="Force preprocessing pipeline (useful if running on raw numpy arrays).")
    parser.add_argument("--no-clip", action="store_true",
                        help="Disable clipping to [0,1] in compatibility mode during raw preprocessing.")

    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    try:
        model = load_model(args.model, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Running prediction on: {args.image}")
    try:
        res = predict(
            model=model,
            image_input=args.image,
            device=args.device,
            preprocess_raw=args.preprocess,
            clip_compatibility=not args.no_clip
        )
        print("\n--- Prediction Results ---")
        pprint.pprint(res)
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()