"""Interactive brain tumor classification dashboard.

Run from the repository root:

    pip install streamlit
    streamlit run dashboard/app.py

Upload an MRI image, get a prediction from the strongest available model, a
Grad-CAM heatmap, per-class probabilities and cross-model rankings.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.inference_utils import (  # noqa: E402
    available_models,
    display_image,
    load_uploaded_image,
    predict_with_model,
    prepare_for_model,
    strongest_model,
)

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_model(arch: str, path: str, device: str):
    from src.inference.predict import load_model

    return load_model(path, arch=arch, device=device)


GRADCAM_AVAILABLE = True
try:
    import pytorch_grad_cam  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    GRADCAM_AVAILABLE = False


def render_model_table(models: dict) -> None:
    rows = []
    for arch, info in models.items():
        rows.append(
            {
                "model": arch,
                "cv_accuracy": round(info["accuracy"], 4) if info["accuracy"] else None,
                "checkpoint_fold": info["fold"],
                "checkpoint": str(info["path"]),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        "cv_accuracy", ascending=False, na_position="last"
    )
    st.dataframe(table, width="stretch", hide_index=True)


def main() -> None:
    st.title("🧠 Brain Tumor Classification Dashboard")
    st.caption(
        "Upload an MRI scan. The strongest model predicts the tumor type and "
        "explains its decision with Grad-CAM."
    )

    models = available_models()
    if not models:
        st.error(
            "No trained checkpoints found. Train models first (e.g. "
            "`./scripts/run_all_training.sh`) or add checkpoints under "
            "`brain_tumor_results/checkpoints/`."
        )
        return

    ranked = sorted(
        models.items(),
        key=lambda item: (item[1]["accuracy"] is None, -(item[1]["accuracy"] or 0.0)),
    )
    default_model = strongest_model(models)

    with st.sidebar:
        st.header("Model")
        # The dashboard always predicts with the strongest model; every other
        # architecture is still shown in the "all architectures" table below.
        arch = default_model
        info = models[arch]
        st.write(f"**Predicting with:** `{arch}` (strongest)")
        st.write(f"**Checkpoint:** `fold {info['fold']}`")
        if info["accuracy"] is not None:
            st.metric("5-fold CV accuracy", f"{info['accuracy'] * 100:.2f}%")
        device = st.radio("Device", ["cpu", "cuda"], index=0, horizontal=True)
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                st.warning("CUDA is not available; falling back to CPU.")
                device = "cpu"

        st.divider()
        cam_method = st.selectbox(
            "Explanation method",
            ["auto", "gradcam++", "gradcam", "xgradcam", "hirescam", "layercam"],
            index=0,
            help=(
                "auto picks the empirically best method per architecture. "
                "VGG19 localizes poorly regardless of method."
            ),
        )
        with st.expander("Available models", expanded=False):
            render_model_table(models)
        st.caption(
            "Grad-CAM: " + ("available" if GRADCAM_AVAILABLE else "not installed")
        )

    uploaded = st.file_uploader(
        "Upload an MRI image (.png, .jpg, .jpeg, .npy)",
        type=["png", "jpg", "jpeg", "npy"],
    )

    if uploaded is None:
        st.info("Upload an image to run a prediction.")
        return

    file_bytes = uploaded.getvalue()
    raw = load_uploaded_image(file_bytes, uploaded.name)
    image, source = prepare_for_model(raw)

    col_input, col_pred = st.columns([1, 1])
    with col_input:
        st.subheader("Input")
        st.image(display_image(image), caption=source, width="stretch")

    with col_pred:
        st.subheader(f"Prediction — {arch} (strongest)")
        model = _load_model(arch, str(info["path"]), device)
        result = predict_with_model(
            model, arch, image, device=device, checkpoint=str(info["path"])
        )
        st.success(
            f"**{result['predicted_class'].title()}** "
            f"(confidence {max(result['probabilities']) * 100:.1f}%)"
        )
        probabilities = pd.DataFrame(
            {
                "class": result["classes"],
                "probability": result["probabilities"],
            }
        ).set_index("class")
        st.bar_chart(probabilities)

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Grad-CAM explainability")
        if not GRADCAM_AVAILABLE:
            st.warning("Install `pytorch-grad-cam` to enable Grad-CAM.")
        else:
            try:
                from dashboard.inference_utils import gradcam_overlay

                overlay = gradcam_overlay(
                    model, arch, image, device=device, method=cam_method
                )
                st.image(
                    overlay,
                    caption=(
                        f"{cam_method} for predicted class: "
                        f"{result['predicted_class']}"
                    ),
                    width="stretch",
                )
            except Exception as exc:  # pragma: no cover
                st.warning(f"Grad-CAM failed for this architecture: {exc}")

    with right:
        st.subheader("Class ranking (worst → best)")
        ranking = pd.DataFrame(result["ranking"])
        ranking["probability"] = ranking["probability"].map(lambda value: f"{value * 100:.1f}%")
        ranking = ranking.rename(
            columns={"class": "class", "probability": "probability", "label": "label"}
        )
        st.dataframe(ranking, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Predictions from all architectures (weakest → strongest by CV accuracy)")
    st.caption(
        "The dashboard always predicts with the strongest model (left panel), but "
        "every available fine-tuned CNN is run on the uploaded image for comparison. "
        "Transfer-learning (features + SVM) models are omitted because their SVM "
        "artifacts are not saved by the training pipeline."
    )

    model_rows = []
    progress = st.progress(0.0, text="Running all architectures...")
    model_items = list(ranked)
    classes = result["classes"]
    for index, (name, model_info) in enumerate(model_items):
        try:
            each_model = _load_model(name, str(model_info["path"]), device)
            each_result = predict_with_model(
                each_model, name, image, device=device, checkpoint=str(model_info["path"])
            )
            row = {
                "model": name,
                "cv_accuracy": model_info["accuracy"],
                "prediction": each_result["predicted_class"],
                "confidence": float(max(each_result["probabilities"])),
                "agrees": each_result["predicted_class"] == result["predicted_class"],
            }
            for class_name, probability in zip(
                each_result["classes"], each_result["probabilities"]
            ):
                row[f"p_{class_name}"] = float(probability)
            model_rows.append(row)
        except Exception as exc:  # pragma: no cover
            model_rows.append(
                {
                    "model": name,
                    "cv_accuracy": model_info["accuracy"],
                    "prediction": f"error: {exc}",
                    "confidence": None,
                    "agrees": False,
                }
            )
        progress.progress((index + 1) / len(model_items), text=f"Ran {name}")
    progress.empty()

    model_table = pd.DataFrame(model_rows).sort_values(
        "cv_accuracy", ascending=True, na_position="first"
    )
    for class_name in classes:
        column = f"p_{class_name}"
        if column in model_table:
            model_table[column] = model_table[column].map(
                lambda value: f"{value * 100:.1f}%" if isinstance(value, float) else value
            )
    model_table = model_table.rename(
        columns={
            **{f"p_{class_name}": f"P({class_name})" for class_name in classes},
            "agrees": f"agrees_with_{arch}",
        }
    )
    model_table["cv_accuracy"] = model_table["cv_accuracy"].map(
        lambda value: f"{value * 100:.2f}%" if value is not None else "n/a"
    )
    model_table["confidence"] = model_table["confidence"].map(
        lambda value: f"{value * 100:.1f}%" if value is not None else "n/a"
    )
    st.dataframe(model_table, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
