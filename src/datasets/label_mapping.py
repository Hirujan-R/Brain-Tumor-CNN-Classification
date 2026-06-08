"""
Central label mapping shared by training, evaluation, and inference.

The original Figshare labels are:
1 = meningioma
2 = glioma
3 = pituitary tumor

Model labels are zero-based and deterministic:
0 = glioma
1 = meningioma
2 = pituitary
"""

ORIGINAL_LABEL_TO_CLASS_NAME = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}

CLASS_NAME_TO_MODEL_LABEL = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
}

MODEL_LABEL_TO_CLASS_NAME = {
    model_label: class_name
    for class_name, model_label in CLASS_NAME_TO_MODEL_LABEL.items()
}

ORIGINAL_LABEL_TO_MODEL_LABEL = {
    original_label: CLASS_NAME_TO_MODEL_LABEL[class_name]
    for original_label, class_name in ORIGINAL_LABEL_TO_CLASS_NAME.items()
}

MODEL_LABEL_TO_ORIGINAL_LABEL = {
    model_label: original_label
    for original_label, model_label in ORIGINAL_LABEL_TO_MODEL_LABEL.items()
}

VALID_ORIGINAL_LABELS = frozenset(ORIGINAL_LABEL_TO_MODEL_LABEL)
VALID_MODEL_LABELS = frozenset(MODEL_LABEL_TO_CLASS_NAME)


def encode_label(original_label: int) -> int:
    original_label = int(original_label)
    if original_label not in ORIGINAL_LABEL_TO_MODEL_LABEL:
        raise ValueError(f"Unknown original label: {original_label}")
    return ORIGINAL_LABEL_TO_MODEL_LABEL[original_label]


def decode_label(model_label: int) -> str:
    model_label = int(model_label)
    if model_label not in MODEL_LABEL_TO_CLASS_NAME:
        raise ValueError(f"Unknown model label: {model_label}")
    return MODEL_LABEL_TO_CLASS_NAME[model_label]


def original_label_name(original_label: int) -> str:
    original_label = int(original_label)
    if original_label not in ORIGINAL_LABEL_TO_CLASS_NAME:
        raise ValueError(f"Unknown original label: {original_label}")
    return ORIGINAL_LABEL_TO_CLASS_NAME[original_label]
