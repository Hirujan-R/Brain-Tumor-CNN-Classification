# Brain Tumor Classification Using Deep Learning and Transfer Learning

<img width="224" height="224" alt="000500" src="https://github.com/user-attachments/assets/ed6fcae7-a56b-46ee-8b10-3cd932f6a580" />
<img width="224" height="224" alt="000111" src="https://github.com/user-attachments/assets/c847865a-3882-489f-a60e-7207c0256bb2" />
<img width="224" height="224" alt="001111" src="https://github.com/user-attachments/assets/de3790fa-9dca-493c-ac90-96fa40958b8d" />


## Overview

This project investigates the effectiveness of multiple deep learning architectures for brain tumor classification from MRI images. The study compares end-to-end fine-tuned Convolutional Neural Networks (CNNs) against transfer learning approaches that use pretrained CNNs as feature extractors combined with Support Vector Machine (SVM) classifiers.

The project follows modern MLOps practices including experiment tracking, data versioning, cloud storage, and reproducible training pipelines.

## Features

* Multiple CNN architecture comparison
* Transfer learning using CNN feature extraction + SVM
* End-to-end CNN fine-tuning
* Experiment tracking with MLflow
* Data and model versioning with DVC
* Amazon S3 artifact storage
* Training on Google Colab
* Model explainability with Grad-CAM
* Reproducible machine learning workflows

---

## Dataset

The dataset consists of brain MRI scans belonging to three classes:

* Glioma
* Meningioma
* Pituitary Tumor

---

## MLOps Architecture

```text
GitHub
   │
   ▼
DVC
   │
   ▼
Amazon S3
   │
   ▼
Google Colab
   │
   ▼
MLflow
   │
   ▼
Model Registry
   │
   ▼
Deployment
```

---

## Models Evaluated

### CNN Architectures

* GoogLeNet
* ResNet18
* VGG19

### Transfer Learning + SVM Architectures

For each architecture:

```text
MRI Image
    │
    ▼
Pretrained CNN
    │
    ▼
Feature Extraction
    │
    ▼
SVM Classifier
    │
    ▼
Prediction
```

Examples:

* GoogLeNet + SVM
* ResNet18 + SVM
* VGG19 + SVM

---

## Tech Stack

| Category                   | Technology          |
| -------------------------- | ------------------- |
| Deep Learning              | PyTorch             |
| Transfer Learning          | Pretrained CNNs     |
| Classical Machine Learning | Scikit-Learn SVM    |
| Experiment Tracking        | MLflow              |
| Data Versioning            | DVC                 |
| Cloud Storage              | Amazon S3           |
| Training Environment       | Google Colab        |
| Explainability             | Grad-CAM            |
| Version Control            | Git, GitHub         |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<username>/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2. Create a Virtual Environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure AWS Credentials

```bash
aws configure
```

Enter:

```text
AWS Access Key ID
AWS Secret Access Key
Default Region
```

### 5. Pull Dataset and Artifacts

```bash
dvc pull
```

---

## Training

Train a model using:

```bash
python -m src.pipelines.train
```

---

## Experiment Tracking

Launch the MLflow UI:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

MLflow tracks:

* Hyperparameters
* Training metrics
* Validation metrics
* Confusion matrices
* ROC curves
* Model artifacts
* Training duration

---

## DVC Workflow

Pull data and artifacts:

```bash
dvc pull
```

Track new datasets:

```bash
dvc add data/
```

Push datasets and artifacts to S3:

```bash
dvc push
```

---

## Results

All models are evaluated with **patient-level 5-fold cross-validation** using the
official `cvind.mat` folds (identical folds for every model), so the comparisons are
paired. Values are **mean ± std across the 5 folds** (3,064 MRI images total).

### End-to-End Fine-Tuned CNNs

| Model                  | Accuracy          | Balanced Accuracy | Precision         | Recall            | F1 (macro)        | ROC-AUC           |
| ---------------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| ResNet18               | **0.9627 ± 0.020** | 0.9613 ± 0.023   | 0.9638 ± 0.020    | **0.9627 ± 0.020** | **0.9587 ± 0.021** | **0.9948 ± 0.005** |
| GoogLeNet              | 0.9607 ± 0.021    | 0.9591 ± 0.021   | 0.9616 ± 0.020    | 0.9607 ± 0.021    | 0.9569 ± 0.021    | 0.9943 ± 0.005    |
| VGG19                  | 0.9598 ± 0.016    | 0.9556 ± 0.013   | 0.9604 ± 0.015    | 0.9598 ± 0.016    | 0.9558 ± 0.015    | 0.9903 ± 0.005    |
| CNN Baseline (scratch) | 0.9176 ± 0.020    | 0.9083 ± 0.021   | 0.9181 ± 0.021    | 0.9176 ± 0.020    | 0.9073 ± 0.022    | 0.9809 ± 0.012    |

### Transfer Learning: Frozen CNN Features + SVM

| Model           | Accuracy       | Balanced Accuracy | Precision      | Recall         | F1 (macro)     | ROC-AUC        |
| --------------- | -------------- | ----------------- | -------------- | -------------- | -------------- | -------------- |
| ResNet18 + SVM  | 0.8962 ± 0.023 | 0.8871 ± 0.024    | 0.8973 ± 0.025 | 0.8962 ± 0.023 | 0.8957 ± 0.024 | 0.9743 ± 0.014 |
| GoogLeNet + SVM | 0.8915 ± 0.023 | 0.8829 ± 0.026    | 0.8936 ± 0.026 | 0.8915 ± 0.023 | 0.8914 ± 0.024 | 0.9720 ± 0.015 |
| VGG19 + SVM     | 0.8797 ± 0.029 | 0.8687 ± 0.029    | 0.8799 ± 0.032 | 0.8797 ± 0.029 | 0.8790 ± 0.031 | 0.9639 ± 0.020 |

### Paired Significance (fine-tuned − transfer, same folds)

| Architecture pair            | Δ Accuracy | Paired t-test (2-sided) | Cohen's d |
| ---------------------------- | ---------- | ----------------------- | --------- |
| ResNet18 vs ResNet18 + SVM   | +6.6 pp    | p < 0.0001              | 13.05     |
| GoogLeNet vs GoogLeNet + SVM | +6.9 pp    | p = 0.0003              | 5.08      |
| VGG19 vs VGG19 + SVM         | +8.0 pp    | p = 0.0006              | 4.37      |

Accuracy, precision, recall and F1 differences remain significant after Bonferroni
correction across all 18 comparisons. ROC-AUC gains are smaller (+2–3 pp) and only
nominally significant. With only 5 folds the exact sign-flip permutation test and the
Wilcoxon signed-rank test have a minimum two-sided p-value of 0.0625 (a power floor);
the paired t-test and the one-sided permutation test (p = 0.03125) carry the evidence.
Full output: `brain_tumor_results/statistical_tests.csv`.

### Per-class Behaviour

Glioma and pituitary are near ceiling for every model. Almost the entire gap lives in
the **meningioma** class: fine-tuned models reach ~92–94% meningioma recall, while the
frozen-feature + SVM models drop to ~75–79%, frequently confusing meningioma with
glioma and pituitary.

### Result Figures

Generated by `python scripts/analyze_results.py --results-dir brain_tumor_results`
(saved under `brain_tumor_results/figures/`):

* `all_models_accuracy_roc_auc.png` — ranked accuracy and ROC-AUC with error bars
* `architecture_pairs_accuracy.png` — fine-tuned vs transfer per architecture
* `architecture_metric_grid.png` — all metrics per architecture
* `per_fold_accuracy.png` — per-fold accuracy for each pair
* `confusion_matrices.png` — pooled out-of-fold confusion matrices

### Explainability: Grad-CAM Localization

Grad-CAM heatmaps were scored against the ground-truth tumor masks (the raw
`tumorMask` field, warped through the same crop/resize as training) over 50 images
using `scripts/evaluate_gradcam.py`:

| Model        | Pointing accuracy | CAM energy in mask | ROC-AUC (best method) |
| ------------ | ----------------- | ------------------ | --------------------- |
| ResNet18     | 0.22              | 0.051              | 0.925 (Grad-CAM++)    |
| GoogLeNet    | 0.10              | 0.058              | 0.894 (Grad-CAM)      |
| CNN Baseline | 0.08              | 0.069              | 0.725 (Grad-CAM++)    |
| VGG19        | 0.06              | 0.034              | 0.561 (Grad-CAM++)    |

High classification accuracy does **not** imply correct localization: every model puts
only ~3–7% of its CAM mass inside the tumor and its peak pixel lands on the lesion only
6–22% of the time. ResNet18 and GoogLeNet localize best (ROC-AUC ≈ 0.89–0.93), the
from-scratch baseline is weaker (≈ 0.73) and VGG19 is near chance (≈ 0.56). The heatmaps
are broad and architecture-dependent, i.e. these classifiers lean on global context
rather than the lesion.

VGG19 is a clear attribution failure: ~48% of its CAM mass falls **outside the head**
(vs ~16–21% for the other models), and no CAM variant fixes it (Grad-CAM, Grad-CAM++,
XGradCAM, HiResCAM and LayerCAM all give tumor ROC-AUC ≈ 0.50–0.56). Background
occlusion did not change its predictions, so the classifier is not using background as a
shortcut — its explanation is simply unreliable. The target layer for VGG19 was changed
from its final MaxPool (below chance) to the last convolution, and the dashboard
auto-selects the best CAM method per architecture (`gradcam.py`), defaulting to ResNet18
for explanations.

---

## Key Findings

End-to-end fine-tuned CNNs consistently outperform frozen CNN feature extraction + SVM
by **6.6–8.0 accuracy points (7–9% relative)** on every architecture, with the gap
present on all 5 folds and very large effect sizes (Cohen's d 4.4–13.1).

Fine-tuning lets the backbone adapt its representations to the tumor classes, which
mainly fixes meningioma (the hardest class). Notably, even the lightweight
`cnn_baseline` trained from scratch (0.9176) beats all three transfer-learning models,
reinforcing that end-to-end adaptation matters more than which pretrained architecture
is used.

The differences *between* the three fine-tuned CNNs (0.960–0.963) and *between* the
three feature+SVM models (0.880–0.896) are small and within fold-level noise, so the
adaptation strategy dominates the architecture choice.

Frozen features + SVM remains attractive when compute is constrained because training
is far cheaper (CNN weights are frozen and a kernel SVM is fit once), but it costs
roughly 7–8 accuracy points.

### Performance Trade-Off

| Approach                     | Advantages                                                          | Disadvantages                        |
| ---------------------------- | ------------------------------------------------------------------- | ------------------------------------ |
| Fine-Tuned CNNs              | Highest accuracy/F1, task-specific features, Grad-CAM explainability | Longer training, higher compute     |
| CNN Feature Extraction + SVM | Much faster training, low compute, strong ROC-AUC                   | ~7–8 pp lower accuracy, no Grad-CAM |

### Conclusion

Fine-tuned CNNs (best: **ResNet18 — 96.3% accuracy, 0.995 ROC-AUC**) are the strongest
models and are statistically significantly better than their transfer-learning
counterparts. Frozen-feature + SVM (~88–90%) is a fast, lower-cost alternative when
compute is limited.

---

## Interactive Dashboard

An interactive Streamlit dashboard lets you upload an MRI image and get a prediction
from the strongest model (**ResNet18**, locked as the prediction model), a Grad-CAM
heatmap, a per-class probability breakdown, and a comparison table showing the
prediction and class probabilities from **all** fine-tuned architectures.

```bash
pip install streamlit
streamlit run dashboard/app.py
```

The app automatically discovers every trained checkpoint under
`brain_tumor_results/checkpoints/` or `reports/checkpoints/`, so it uses whichever
models you trained (with a fallback to `model/model.pth`). Grad-CAM is available for
the fine-tuned CNNs; the feature-extraction + SVM models are shown as a comparison
ranking when available.

---

## Future Improvements

* Docker Deployment
* FastAPI Inference Service
* CI/CD Pipeline
* Automated Model Retraining
* Persist SVM feature-extractor + classifier artifacts for dashboard use

---

## Author

**Hirujan Rangaraj**

Brain Tumor Classification using Deep Learning, Transfer Learning, and MLOps.
