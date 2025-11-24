# 🎤 Presentation Materials — Diabetes Prediction AI Project

This folder contains the presentation assets summarizing the full workflow of the Diabetes Prediction AI Project, including the methodology, key findings, model comparisons, and final evaluation visuals. These materials are intended for use in class presentations, video walkthroughs, or project documentation.

---

#  Methodology Overview

The methodology used in this project follows a complete machine-learning lifecycle, from data ingestion to final model evaluation. The process is broken into clear, reproducible phases:

---

##  Data Acquisition & Understanding
- Loaded the dataset from `data/diabetes_prediction_dataset.csv`
- Inspected columns, data types, distributions, and label balance
- Identified numeric and categorical features
- Verified dataset quality using:
  - Missing value checks
  - Duplicate analysis
  - Outlier detection
  - Correlation analysis

_Notebooks used: `data_quality_report.ipynb`, `feature_analysis.ipynb`_

---

##  Data Preprocessing
Performed in `src/preprocessing.py`:

- Applied **70/15/15 stratified split**:
  - 70% training  
  - 15% validation  
  - 15% test  
- Imputed missing values:
  - Mean for numeric
  - `"Unknown"` for categorical
- Built a **ColumnTransformer**:
  - `StandardScaler` → numeric features  
  - `OneHotEncoder` → categorical features  
- Fitted the preprocessor on the training split  
- Transformed train/validation/test sets into model-ready NumPy arrays

---

##  Model Development
Performed in `src/train.py`.

Two models were trained:

###  XGBoost Classifier
- 300 trees  
- Max depth = 4  
- Learning rate = 0.05  
- Highly effective on mixed numeric/categorical engineered features  

###  Neural Network (MLPClassifier)
- 2 hidden layers: (64, 32)
- ReLU activation
- Early stopping enabled
- Works well with scaled + one-hot encoded data

Both models were trained on the processed training data.  
Validation metrics were computed to monitor overfitting.

Artifacts saved:
- `preprocessor.pkl`
- `xgboost.pkl`
- `neural_network.pkl`

---

##  Model Evaluation
Performed in `src/evaluate.py`.

Steps:
- Reloaded raw dataset
- Re-created the exact train/val/test split
- Loaded saved preprocessor + models
- Transformed only **test set**
- Computed final performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

Generated visuals:
- Confusion Matrices  
- ROC Curves  

These images are saved in the `results/` folder and used for presentation.

---

##  Results Summary

###  XGBoost
- **AUC ≈ 0.979**
- Strong separation between classes  
- Lowest false positive rate  

###  Neural Network
- **AUC ≈ 0.976**
- Slightly higher misclassifications  
- Still excellent performance  

Conclusion:  
**XGBoost is the best performing model overall**, but both models achieve high accuracy and generalization.

---

#  Files in This Folder

This folder may contain:
- Slide deck(s) for class presentation  
- PDF or PowerPoint versions of the talk  
- Additional diagrams or exported images  
- Notes or scripts for presenting the methodology  

---

# 🗂 Related Project Components

- **Preprocessing Code:** `src/preprocessing.py`  
- **Training Pipeline:** `src/train.py`  
- **Evaluation Scripts:** `src/evaluate.py`  
- **Results & Plots:** `results/`  
- **Exploration Notebooks:** `notebooks/`

---

#  Summary

This README provides a structured methodology overview suitable for:
- Presentations  
- Documentation  
- Oral exams  
- Walkthrough videos  
- Portfolio submission  

The `presentation/` folder acts as the final consolidated space for sharing the story of the project: problem → methods → models → results → conclusions.

