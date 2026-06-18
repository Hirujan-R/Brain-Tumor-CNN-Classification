import torch
import torch.nn as nn

import numpy as np
from torchvision.models import resnet18, ResNet18_Weights, googlenet, GoogLeNet_Weights, vgg19, VGG19_Weights
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from base_model import BrainTumorModel


class PretrainedFeatureExtractor(BrainTumorModel):
    """
    Wraps a pretrained PyTorch model and strips the classification head
    to output raw feature embeddings.
    """

    def __init__(self, base_model_name: str = "resnet18"):
        super().__init__()

        if base_model_name.lower() == "resnet18":
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif base_model_name.lower() == "googlenet":
            self.model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif base_model_name.lower() == "vgg16":
            self.model = vgg19(weights=VGG19_Weights.DEFAULT)
            self.feature_dim = 4096
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier.children())[:-1]
            )
        else:
            raise NotImplementedError(
                f"Model {base_model_name} is not implemented."
            )
        self.freeze_all_layers()
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SVMTrainerWrapper:
    """
    Scikit-Learn SVM wrapped in a Trainer-like interface.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        kernel: str = "rbf",
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.feature_extractor = feature_extractor
        self.device = next(feature_extractor.parameters()).device

        self.svm = SVC(
            kernel=kernel,
            C=C,
            random_state=random_state,
            probability=True,
        )

        self.is_fitted = False
        self.history = None

    def _extract_features(
        self,
        dataloader,
    ) -> tuple[np.ndarray, np.ndarray]:

        self.feature_extractor.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():

            for batch in dataloader:

                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                if images.dim() == 3:
                    images = images.unsqueeze(0)

                images = images.to(self.device)

                features = self.feature_extractor(images)

                all_features.append(
                    features.cpu().numpy()
                )

                all_labels.append(
                    labels.detach().cpu().numpy()
                )

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        return X, y

    def fit_svm(self, train_loader) -> None:

        print("Extracting features from training set...")

        X_train, y_train = self._extract_features(
            train_loader
        )

        print(
            f"Training SVM on "
            f"{X_train.shape[0]} samples "
            f"with {X_train.shape[1]} features..."
        )

        self.svm.fit(X_train, y_train)

        self.is_fitted = True

        print("SVM training complete.")

    def evaluate_svm(
        self,
        val_loader,
    ) -> tuple[float, float]:

        if not self.is_fitted:
            raise RuntimeError(
                "SVM must be trained before evaluation."
            )

        X_val, y_val = self._extract_features(
            val_loader
        )

        preds = self.svm.predict(X_val)
        probs = self.svm.predict_proba(X_val)

        accuracy = accuracy_score(
            y_val,
            preds,
        )

        f1 = f1_score(
            y_val,
            preds,
            average="weighted",
        )


        precision = precision_score(y_val, preds, average="weighted")
        recall = recall_score(y_val, preds, average="weighted")

        roc_auc = roc_auc_score(
            y_val,
            probs,
            multi_class="ovr",   # one-vs-rest
            average="weighted"
        )

        print(
            f"Validation Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        return accuracy, f1, precision, recall, roc_auc, preds, y_val

    def fit(
        self,
        train_loader,
        val_loader,
    ):

        print(
            "\n[SVM PIPELINE] "
            "Starting feature extraction + training"
        )

        self.fit_svm(train_loader)

        val_accuracy, val_f1, val_precision, val_recall, val_roc_auc, preds, y_val = self.evaluate_svm(
            val_loader
        )

        self.history = {
            "val_accuracy": [val_accuracy],
            "val_f1": [val_f1],
            "val_precision": [val_precision],
            "val_recall": [val_recall], 
            "val_roc_auc": [val_roc_auc],
            "val_loss": False,
            "targets": y_val.flatten(),
            "preds": preds.flatten()
        }

        print(
            f"SVM Fit Complete | "
            f"Acc: {val_accuracy:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        return self.history

    def get_best_metrics(self):

        if self.history is None:
            raise RuntimeError(
                "Model not trained yet."
            )

        return {
            "accuracy": self.history["val_accuracy"][0],
            "f1": self.history["val_f1"][0],
        }

    def predict(
        self,
        images: torch.Tensor,
    ) -> np.ndarray:

        if not self.is_fitted:
            raise RuntimeError(
                "SVM must be trained before prediction."
            )

        self.feature_extractor.eval()

        with torch.no_grad():

            images = images.to(self.device)

            features = (
                self.feature_extractor(images)
                .cpu()
                .numpy()
            )

        return self.svm.predict(features)