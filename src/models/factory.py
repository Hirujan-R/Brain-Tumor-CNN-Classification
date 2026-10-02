"""Central model registry / factory.

All training pipelines (fine-tuning and CNN-feature-extraction + SVM) build
their models through these helpers so that model names stay consistent across
the codebase, notebooks, and CLIs.
"""

try:
    from .cnn_baseline import CNNBaseline
    from .googlenet import GoogLeNetBrainTumor
    from .finetune_models import ResNet18BrainTumor, VGG19BrainTumor
    from .transfer_models import PretrainedFeatureExtractor
except ImportError:  # allows running with src/models on sys.path
    from cnn_baseline import CNNBaseline
    from googlenet import GoogLeNetBrainTumor
    from finetune_models import ResNet18BrainTumor, VGG19BrainTumor
    from transfer_models import PretrainedFeatureExtractor


# End-to-end fine-tuned classifiers
FINE_TUNE_MODELS = (
    "cnn_baseline",
    "googlenet",
    "resnet18",
    "vgg19",
)

# Frozen pretrained CNN feature extractor + SVM classifier
TRANSFER_SVM_MODELS = (
    "googlenet+svm",
    "resnet18+svm",
    "vgg19+svm",
)

ALL_MODELS = FINE_TUNE_MODELS + TRANSFER_SVM_MODELS


def is_svm_model(model_name: str) -> bool:
    return model_name in TRANSFER_SVM_MODELS


def base_arch(model_name: str) -> str:
    """'resnet18+svm' -> 'resnet18'."""
    return model_name.replace("+svm", "")


def build_classifier(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = True,
    aux_logits: bool = True,
):
    """Build an end-to-end trainable classifier."""
    model_name = base_arch(model_name)

    if model_name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes)
    if model_name == "googlenet":
        return GoogLeNetBrainTumor(
            num_classes=num_classes, pretrained=pretrained, aux_logits=aux_logits
        )
    if model_name == "resnet18":
        return ResNet18BrainTumor(num_classes=num_classes, pretrained=pretrained)
    if model_name == "vgg19":
        return VGG19BrainTumor(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown classifier: {model_name}")


def build_feature_extractor(model_name: str) -> PretrainedFeatureExtractor:
    """Build a frozen pretrained CNN feature extractor for SVM training."""
    arch = base_arch(model_name)
    if arch not in {"cnn_baseline", "googlenet", "resnet18", "vgg19"}:
        raise ValueError(f"Unknown feature extractor: {model_name}")
    if arch == "cnn_baseline":
        raise ValueError("cnn_baseline has no pretrained weights for transfer learning.")
    return PretrainedFeatureExtractor(base_model_name=arch)


def build_model(model_name: str, **kwargs):
    """Build either a classifier or a feature extractor based on the name."""
    if is_svm_model(model_name):
        return build_feature_extractor(model_name)
    return build_classifier(model_name, **kwargs)
