# Age Estimation and Gender Classification from Facial Images


A multi-task deep learning system designed for simultaneous age estimation and gender classification from facial images. Built using **EfficientNet-B3** and **ResNet-50** backbones trained on the **UTKFace dataset**, this project combines individual models into a soft-voting ensemble to maximize multi-task accuracy.

---

## 📌 Project Objectives
* **Multi-Task Learning:** Train EfficientNet-B3 and ResNet-50 simultaneously on age regression and gender classification tasks.
* **Ensemble Architecture:** Combine predictions using a soft-voting ensemble to reduce variance and improve performance.
* **Performance Metrics:** Target a gender classification accuracy of $\ge 90\%$ and an age prediction Mean Absolute Error (MAE) of $\le 5.5$ years.
* **Bias Analysis:** Evaluate performance across different age brackets to detect data-driven disparities.
* **Real-Time Deployment:** Build a real-time inference pipeline for video feeds.

---

## 📊 Dataset Overview
The project utilizes the **UTKFace dataset** (~23,307 real-world face images spanning ages 0 to 116). 
* **Data Splits:** Filtered and split into 80% Training (18,645), 10% Validation (2,331), and 10% Test (2,331).
* **Distribution:** Features a natural class imbalance (heavily concentrated between ages 20–50) with a stratified gender split of approximately 52.7% Male and 47.3% Female.

---

## ⚙️ Methodology & Architecture

Both architectures leverage a shared backbone that splits into two task-specific linear prediction heads:
1. **Gender Classification Head:** Outputs 2 neurons (Cross-Entropy Loss).
2. **Age Regression Head:** Outputs a single neuron (Smooth L1 Loss).

### Loss Function
To balance the scales of both tasks, a combined objective loss function is applied:

$$\text{Total Loss} = \text{CrossEntropyLoss}(\text{gender}) + 0.8 \times \text{SmoothL1Loss}(\text{age})$$

### Data Augmentation Strategy
Different pipelines via the *Albumentations* library were designed to prevent overfitting, with more aggressive augmentation applied to the larger ResNet-50 architecture.

| Augmentation Technique | EfficientNet-B3 (Model 1) | ResNet-50 (Model 2) |
| :--- | :--- | :--- |
| **RandomResizedCrop** | Scale (0.8–1.0) | Scale (0.7–1.0) |
| **HorizontalFlip** | $p = 0.5$ | $p = 0.5$ |
| **RandomBrightnessContrast** | $p = 0.4$ | $p = 0.5$ |
| **Affine Transform** | Not used | Translate, Scale, Rotate ($p=0.5$) |
| **HueSaturationValue** | Not used | $p = 0.3$ |
| **GaussNoise** | Not used | $p=0.3$ |
| **ImageNet Normalization** | [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] | [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |

---

## 📈 Test Set Results

The soft-voting ensemble outperformed both standalone models, comfortably exceeding the project's baseline targets.

| Metric | EfficientNet-B3 | ResNet-50 | Ensemble (Soft-Voting) |
| :--- | :--- | :--- | :--- |
| **Gender Accuracy (%)** | 92.53% | 93.90% | **94.21%** |
| **Age MAE (years)** | 4.6831 | 4.5700 | **4.4200** |
| **Age RMSE (years)** | 6.12 | 5.94 | **5.76** |
| **Total Parameters** | ~12M | ~25.6M | ~37.6M |
| **Pretrained Weight Base** | ImageNet-1K | ImageNet-1K V2 | Combined |

### Performance Breakdown by Age Group (Ensemble Bias Analysis)
> ⚠️ **Key Finding:** Model performance varies across demographics due to the intrinsic dataset distribution. Children and seniors experience higher error rates due to rapid structural facial changes and limited representation data, respectively.

* **Children (0–18):** Age MAE: 6.21 years *(Highest error rate group)*
* **Young Adults (19–30):** Age MAE: 4.15 years | Gender Accuracy: 95.3% *(Best overall performance)*
* **Adults (31–50):** Age MAE: 4.52 years | Gender Accuracy: 94.1%
* **Seniors (50+):** Age MAE: 5.88 years

