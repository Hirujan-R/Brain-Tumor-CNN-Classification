import torch
from src.inference.predict import load_model, predict
import numpy as np

def batch_predict(image_tensors, model_path="model/model.pth", num_classes=3, device="cpu"):
    model = load_model(model_path=model_path, num_classes=num_classes, device=device)
    predictions = []
    for image_tensor in image_tensors:
        predictions.append(predict(model, image_tensor, device=device))
    return predictions




