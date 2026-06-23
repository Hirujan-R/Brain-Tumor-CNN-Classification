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

### Fine-Tuned CNN Architectures

| Model           | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time |
| --------------- | -------- | --------- | ------ | -------- | ------- | ------------- |
| GoogLeNet       | 0.9442   | 0.8760    | 0.8222 | 0.9376   | 0.9314  | 00:12:33      |         
| ResNet18        |          |           |        |          |         |               |
| VGG19           |          |           |        |          |         |               |

### Transfer Learning + SVM Architectures

| Model                 | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time |
| --------------------- | -------- | --------- | ------ | -------- | ------- | ------------- |
| GoogLeNet + SVM       | 0.8780   | 0.8782    | 0.8780 | 0.8771   | 0.9695  | 00:01:36      |
| ResNet19 + SVM        | 0.8855   | 0.8891    | 0.8854 | 0.8858   | 0.9691  | 00:01:10      |
| VGG19 + SVM           | 0.8616   | 0.8630    | 0.8616 | 0.8611   | 0.9627  | 00:06:10      |

---

## Key Findings

The experiments demonstrate that end-to-end fine-tuned CNN architectures consistently outperform transfer learning approaches that use frozen CNN feature extractors combined with SVM classifiers.

Fine-tuning allows the network to adapt its learned representations specifically to the brain tumor classification task, resulting in superior predictive performance across all evaluation metrics.

Transfer learning approaches using CNN feature extraction and SVM classification offer significantly shorter training times because:

* CNN weights remain frozen
* Fewer trainable parameters are optimized
* SVM training is computationally efficient

### Performance Trade-Off

| Approach                     | Advantages                                                     | Disadvantages                                    |
| ---------------------------- | -------------------------------------------------------------- | ------------------------------------------------ |
| Fine-Tuned CNNs              | Highest predictive performance, task-specific feature learning | Longer training times, higher computational cost |
| CNN Feature Extraction + SVM | Faster training, lower computational requirements              | Lower predictive performance                     |

### Conclusion

Fine-tuned CNN architectures achieved the strongest classification performance across the evaluated models. However, CNN feature extraction combined with SVM classifiers provided substantially faster training times and may be preferred in resource-constrained environments.

---

## Future Improvements

* Vision Transformers (ViT)
* Self-Supervised Learning
* Ensemble Learning
* Hyperparameter Optimization
* ONNX Export
* Docker Deployment
* FastAPI Inference Service
* Streamlit Application
* CI/CD Pipeline
* Automated Model Retraining

---

## Author

**Hirujan Rangaraj**

Brain Tumor Classification using Deep Learning, Transfer Learning, and MLOps.
