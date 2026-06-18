import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import numpy as np

# Adjust imports based on your package structure

from cnn_baseline import CNNBaseline
from googlenet import GoogLeNetBrainTumor
from transfer_models import PretrainedFeatureExtractor, SVMTrainerWrapper

def test_cnn_baseline():
    print("Testing CNNBaseline...")
    model = CNNBaseline(num_classes=3)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"
    print(f"[OK] CNNBaseline output shape: {output.shape}")
    print(f"[OK] CNNBaseline parameters: {model.count_parameters()}")

def test_googlenet():
    print("\nTesting GoogLeNetBrainTumor (training mode with aux_logits)...")
    # Training mode
    model = GoogLeNetBrainTumor(num_classes=3, pretrained=False, aux_logits=True)
    model.train()
    dummy_input = torch.randn(4, 3, 224, 224)
    outputs = model(dummy_input)
    # outputs is a namedtuple: (logits, aux_logits2, aux_logits1)
    # For some torchvision versions it might return a tensor if aux_logits=False, 
    # but with aux_logits=True it returns GoogLeNetOutputs
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    assert logits.shape == (4, 3), f"Expected (4, 3), got {logits.shape}"
    print(f"[OK] GoogLeNet (train) output shape: {logits.shape}")
    
    print("\nTesting GoogLeNetBrainTumor (eval mode)...")
    model.eval()
    outputs_eval = model(dummy_input)
    assert outputs_eval.shape == (4, 3), f"Expected (4, 3), got {outputs_eval.shape}"
    print(f"[OK] GoogLeNet (eval) output shape: {outputs_eval.shape}")
    print(f"[OK] GoogLeNet parameters: {model.count_parameters()}")

def test_transfer_models(base_model_name):
    print("\nTesting PretrainedFeatureExtractor & SVMTrainerWrapper...")
    extractor = PretrainedFeatureExtractor(base_model_name=base_model_name)
    
    # Check freezing
    trainable_params = extractor.count_parameters()
    assert trainable_params == 0, f"Expected 0 trainable params, got {trainable_params}"
    
    dummy_input = torch.randn(4, 3, 224, 224)
    features = extractor(dummy_input)
    if base_model_name == "resnet18":
        expected_dim = 512
    elif base_model_name == "googlenet":
        expected_dim = 1024
    elif base_model_name in ["vgg16", "vgg19"]:
        expected_dim = 4096

    assert features.shape == (4, expected_dim), f"Expected {expected_dim}, got {features.shape}"
    print(f"[OK] PretrainedFeatureExtractor output shape: {features.shape}")

    print("\nTesting SVMTrainerWrapper with dummy DataLoader...")
    
    # Create a dummy dataloader
    class DummyDataset:
        def __init__(self):
            self.data = [(torch.randn(3, 224, 224), torch.tensor(np.random.randint(0, 3))) for _ in range(16)]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
            
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    
    svm_wrapper = SVMTrainerWrapper(feature_extractor=extractor)
    
    history = svm_wrapper.fit(dummy_loader, dummy_loader)
    print(f"[OK] SVMTrainerWrapper evaluated successfully with acc {history["val_accuracy"][0]:.2f}")

if __name__ == "__main__":
    test_cnn_baseline()
    test_googlenet()
    test_transfer_models(base_model_name="googlenet")
    print("\nAll model architecture tests passed!")
