import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from src.inference.predict import load_model


def generate_gradcam(
    model,
    image_tensor,
    target_class=None,
    device="cpu"
):

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

    target_layer = model.inception5b

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    if target_class is None:
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

            pred_class = torch.argmax(probs, dim=1).item()
            target_class = pred_class

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)]
    )

    grayscale_cam = grayscale_cam[0]

    return grayscale_cam, probs, pred_class


def visualize_gradcam(image_np, cam):
    image = image_np.astype(np.float32)

    image = (image - image.min()) / (
        image.max() - image.min() + 1e-8
    )

    visualization = show_cam_on_image(
        image,
        cam,
        use_rgb=True
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.axis("off")
    plt.show()


def main():
    model = load_model(
        "model/model.pth",
        num_classes=3,
        device="cpu"
    )

    image = np.load(
        "data/processed/images/000896.npy"
    )

    cam, probs, pred_class = generate_gradcam(
        model=model,
        image_tensor=image,
        target_class=None,
        device="cpu"
    )

    print("Predicted Class:", pred_class)
    print("Predicted Probability:", probs)

    visualize_gradcam(image, cam)

if __name__ == "__main__":
    main()