import torch
from src.inference.predict import load_model, predict
import numpy as np

def batch_predict(image_tensors, model_path="model/model.pth", num_classes=3, device="cpu"):
    model = load_model(model_path=model_path, num_classes=num_classes, device=device)
    predictions = []
    for image_tensor in image_tensors:
        predictions.append(predict(model, image_tensor, device=device))
    return predictions

def main():
    image_tensors = [np.load("data/processed/images/000001.npy"), np.load("data/processed/images/000002.npy"), 
                     np.load("data/processed/images/000003.npy"), np.load("data/processed/images/000004.npy")]
    predictions = batch_predict(image_tensors=image_tensors)
    print(predictions)

if __name__ == "__main__":
    main()


