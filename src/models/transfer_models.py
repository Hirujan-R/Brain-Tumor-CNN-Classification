import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from base_model import BrainTumorModel

class PretrainedFeatureExtractor(BrainTumorModel):
    """
    Wraps a pretrained PyTorch model and strips the classification head
    to output raw 1D feature embeddings for downstream tasks (like SVM).
    """
    def __init__(self, base_model_name: str = "resnet18"):
        super().__init__()
        
        if base_model_name.lower() == "resnet18":
            # Load pretrained ResNet18
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            # Remove the final fully connected layer by replacing it with Identity
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        else:
            raise NotImplementedError(f"Model {base_model_name} is not implemented for extraction yet.")
            
        # Freeze all layers since we only use it for feature extraction
        self.freeze_all_layers()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts features.
        Output shape: (batch_size, feature_dim)
        """
        return self.model(x)

class SVMTrainerWrapper:
    """
    A utility class to wrap Scikit-Learn's SVM with a PyTorch DataLoader
    and the PretrainedFeatureExtractor.
    """
    def __init__(self, feature_extractor: nn.Module, kernel: str = 'rbf', C: float = 1.0, random_state: int = 42):
        self.feature_extractor = feature_extractor
        self.device = next(feature_extractor.parameters()).device
        self.svm = SVC(kernel=kernel, C=C, random_state=random_state, probability=True)
        self.is_fitted = False
        
    def _extract_features(self, dataloader) -> tuple[np.ndarray, np.ndarray]:
        """Runs the entire dataloader through the feature extractor to get numpy arrays."""
        self.feature_extractor.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # The Dataset might return (image, label) or (image, label, metadata)
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                    
                images = images.to(self.device)
                
                # Extract features
                features = self.feature_extractor(images)
                
                # Move back to CPU and convert to numpy
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                
        # Concatenate all batches
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        return X, y

    def train(self, train_loader) -> None:
        """Extracts features from the train_loader and fits the SVM."""
        print("Extracting features from training set...")
        X_train, y_train = self._extract_features(train_loader)
        
        print(f"Training SVM on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.svm.fit(X_train, y_train)
        self.is_fitted = True
        print("SVM training complete.")
        
    def evaluate(self, val_loader) -> float:
        """Evaluates the trained SVM on a validation loader."""
        if not self.is_fitted:
            raise RuntimeError("SVM must be trained before calling evaluate().")
            
        print("Extracting features from validation set...")
        X_val, y_val = self._extract_features(val_loader)
        
        predictions = self.svm.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def predict(self, images: torch.Tensor) -> np.ndarray:
        """Predicts classes for a batch of raw image tensors."""
        if not self.is_fitted:
            raise RuntimeError("SVM must be trained before calling predict().")
            
        self.feature_extractor.eval()
        with torch.no_grad():
            images = images.to(self.device)
            features = self.feature_extractor(images).cpu().numpy()
            
        return self.svm.predict(features)
