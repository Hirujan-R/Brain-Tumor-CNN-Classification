import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from src.inference.predict import load_model
from src.datasets.label_mapping import MODEL_LABEL_TO_CLASS_NAME


def generate_gradcam(
    model,
    image_tensor,
    target_class=None,
    device="cpu",
    preprocessed=True
):
    """
    Generate GradCAM visualization for a brain tumor image.
    
    Args:
        model: Loaded GoogLeNet model
        image_tensor: Input image as numpy array (H, W, C)
        target_class: Target class for visualization (None = use prediction)
        device: Device to use
        preprocessed: If True, image is already preprocessed (from .npy)
    """
    model.eval()

    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.tensor(
            image_tensor,
            dtype=torch.float32
        )
    else:
        raise TypeError("image_tensor must be a np array.")

    # HWC -> CHW
    input_tensor = image_tensor.permute(2, 0, 1)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # GoogLeNetBrainTumor wraps the actual GoogLeNet in self.model
    target_layer = model.model.inception5b

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    if target_class is None:
        target_class = pred_class

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)]
    )

    grayscale_cam = grayscale_cam[0]

    return grayscale_cam, probs, pred_class


def visualize_gradcam(image_np, cam, save_path=None):
    image = image_np.astype(np.float32)

    image = (image - image.min()) / (
        image.max() - image.min() + 1e-8
    )

    visualization = show_cam_on_image(
        image,
        cam,
        use_rgb=True
    )

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.imsave(save_path, visualization)
        print(f"Saved GradCAM: {save_path}")
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(visualization)
        plt.axis("off")
        plt.show()


def main():

    import argparse

    parser = argparse.ArgumentParser(description="Generate gradcam of sample from input dataset using ID.")
    parser.add_argument("--sample_id", type=str, help="Sample ID")
    args = parser.parse_args()

    model = load_model(
        "model/best_epoch.pth",
        num_classes=3,
        device="cpu"
    )

    image = np.load(
        f"data/processed/images/{args.sample_id}.npy"
    )

    cam, probs, pred_class = generate_gradcam(
        model=model,
        image_tensor=image,
        target_class=None,
        device="cpu"
    )

    print("Predicted class index (0=glioma, 1=meningioma, 2=pituitary):", pred_class)
    print("Predicted class name:", MODEL_LABEL_TO_CLASS_NAME.get(pred_class, "unknown"))
    print("Probabilities:", probs)
    visualize_gradcam(image, cam, save_path=f"output/{args.sample_id}.png")

if __name__ == "__main__":
    main()