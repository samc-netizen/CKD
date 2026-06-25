# Chronic Kidney Disease Prediction Using Machine Learning

A complete machine learning pipeline that classifies whether a patient has **Chronic Kidney Disease (CKD)** based on clinical and laboratory attributes, built on the UCI Machine Learning Repository's CKD dataset.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

---

## Overview

Chronic Kidney Disease is a progressive condition in which kidney function declines over a period of 3 months or more, often going undiagnosed until later stages because early symptoms (fatigue, swelling, reduced urine output) are subtle. This project builds a **binary classification model** that predicts CKD presence from 24 clinical attributes, enabling earlier detection and intervention.

**Problem Type:** Supervised Binary Classification
**Target Classes:** `1` = CKD present, `0` = CKD not present

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) (mirrored on [Kaggle](https://www.kaggle.com/datasets/akshayksingh/kidney-disease-dataset))
- **Records:** 400 patients
- **Attributes:** 24 features + 1 target column

| Feature | Description | Feature | Description |
|---|---|---|---|
| age | Age (years) | pcc | Pus cell clumps |
| bp | Blood pressure (mm/Hg) | ba | Bacteria |
| sg | Specific gravity | bgr | Blood glucose random |
| al | Albumin | bu | Blood urea |
| su | Sugar | sc | Serum creatinine |
| rbc | Red blood cells | sod | Sodium |
| pc | Pus cell | pot | Potassium |
| hemo | Hemoglobin | pcv | Packed cell volume |
| wc | White blood cell count | rc | Red blood cell count |
| htn | Hypertension | dm | Diabetes mellitus |
| cad | Coronary artery disease | appet | Appetite |
| pe | Pedal edema | ane | Anemia |
| **class** | **Target: ckd / notckd** | | |

---

## Repository Structure

```
CKD-Prediction-Using-Machine-Learning/
│
├── README.md                      # This file
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── dataset/
│   └── kidney_disease.csv          # Raw CKD dataset (400 rows × 25 cols)
│
├── notebooks/
│   └── CKD_Prediction.ipynb        # Full end-to-end ML pipeline notebook
│
├── models/
│   └── final_model.pkl             # Serialized best model + preprocessing artifacts
│
├── reports/
│   └── CKD_Project_Report.md       # Condensed project report (8-12 pages)
│
└── images/
    ├── class_distribution.png
    ├── biomarker_distributions.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── model_comparison.png
    ├── feature_importance.png
    └── cross_validation.png
```

---

## Methodology

```
Dataset Collection → Data Cleaning → Preprocessing → EDA →
Feature Selection → Train-Test Split → Model Training (6 algorithms) →
Model Evaluation → Cross-Validation → Best Model Selection → Save Model
```

**Preprocessing steps:**
- Stripped whitespace inconsistencies in categorical fields
- Imputed missing numeric values using **median**
- Imputed missing categorical values using **mode**
- Label-encoded categorical variables
- Standardized features for distance-based models (Logistic Regression, SVM, KNN)
- 80/20 stratified train-test split

**Models trained:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, K-Nearest Neighbors

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, Specificity, AUC-ROC, Confusion Matrix, 5-Fold Cross-Validation

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | Specificity | AUC-ROC |
|---|---|---|---|---|---|---|
| **Random Forest (selected)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 0.9875 | 0.9804 | 1.0000 | 0.9901 | 0.9667 | 1.0000 |
| SVM | 0.9875 | 0.9804 | 1.0000 | 0.9901 | 0.9667 | 1.0000 |
| Decision Tree | 0.9500 | 0.9792 | 0.9400 | 0.9592 | 0.9667 | 0.9533 |

Three models (Random Forest, Logistic Regression, KNN) tied at a perfect test-set score. **Random Forest** was selected as the final model because 5-fold cross-validation showed it (along with SVM) achieved a perfect, **zero-variance F1 score (1.0000 ± 0.0000)** across all folds — the most stable result of any model — while also providing interpretable feature importances for clinical insight.

> Full comparison table, confusion matrices, ROC curves, and cross-validation plots are in `notebooks/CKD_Prediction.ipynb` (Sections 8–10) and `reports/CKD_Project_Report.md`.

**Top predictive features:** Hemoglobin, Specific Gravity, Serum Creatinine, Packed Cell Volume, Albumin — consistent with established CKD clinical markers.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/CKD-Prediction-Using-Machine-Learning.git
cd CKD-Prediction-Using-Machine-Learning

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/CKD_Prediction.ipynb
```

### Using the saved model for prediction

```python
import pickle
import pandas as pd

with open('models/final_model.pkl', 'rb') as f:
    artifact = pickle.load(f)

# See notebooks/CKD_Prediction.ipynb Section 13 for the full predict_ckd() helper function
```

---

## Future Scope

- Validate on larger, real-world multi-hospital datasets
- Explore deep learning architectures (e.g., feed-forward neural nets, TabNet)
- Build a web application (Flask/Streamlit) for clinician-facing predictions
- Incorporate longitudinal patient data to predict CKD progression stage, not just presence
- Add SHAP/LIME explainability for clinical trust and interpretability

---

## References

1. UCI Machine Learning Repository — Chronic Kidney Disease Dataset
2. Kaggle — [Kidney Disease Dataset](https://www.kaggle.com/datasets/akshayksingh/kidney-disease-dataset)
3. Scikit-learn Documentation — https://scikit-learn.org/
