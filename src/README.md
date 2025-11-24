# Source Code (`src/`) — Diabetes Prediction AI Project

This folder contains all Python source files required to run the full machine-learning pipeline for the Diabetes Prediction AI Project. These scripts perform data preprocessing, model training, evaluation, and utility support functions. Together, they automate the entire workflow from raw CSV data → trained models → performance results.

---

## 📦 Files Included

### 1️⃣ **preprocessing.py**
Handles all data loading, cleaning, splitting, and feature transformation.  
📄 Source: :contentReference[oaicite:0]{index=0}

#### Key Responsibilities
- Load the raw dataset from the `data/` folder  
- Validate that the target column exists  
- Perform a **70/15/15 train–validation–test split** with stratification  
- Identify numeric and categorical columns  
- Impute missing values (mean for numeric, `"Unknown"` for categorical)  
- Build a **ColumnTransformer** using:
  - `StandardScaler` for numeric features  
  - `OneHotEncoder` for categorical features  
- Fit the preprocessor on training data  
- Transform train, validation, and test splits into NumPy arrays  

#### Outputs
- Transformed datasets: `X_train_proc`, `X_val_proc`, `X_test_proc`  
- A **fitted preprocessor object** (saved later as `preprocessor.pkl`)

---

### 2️⃣ **train.py**
End-to-end training script for both selected models: **XGBoost** and **Neural Network (MLPClassifier)**.  
📄 Source: :contentReference[oaicite:1]{index=1}

#### What This Script Does
1. Loads raw CSV data  
2. Creates train/validation/test splits  
3. Fits the preprocessing pipeline  
4. Trains two models:
   - **XGBoostClassifier**  
   - **MLPClassifier (Neural Network)**  
5. Computes validation metrics:
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - ROC-AUC  
6. Saves model artifacts into `results/`:
   - `preprocessor.pkl`  
   - `xgboost.pkl`  
   - `neural_network.pkl`  

#### Custom Model Building Functions
- `build_xgboost_model()`  
- `build_neural_net(input_dim)`  

#### Notes
- Uses early stopping for the neural network  
- Uses robust defaults for XGBoost hyperparameters  
- Implements a consistent random seed for reproducibility  

---

### 3️⃣ **evaluate.py**
Evaluates saved models on the **held-out test set** and generates all final analysis plots.  
📄 Source: :contentReference[oaicite:2]{index=2}

#### Pipeline
1. Load raw data  
2. Recreate train/val/test split  
3. Load saved preprocessor + models from `results/`  
4. Transform test data  
5. Compute full test metrics:
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - ROC-AUC  
6. Generate and save:
   - 🖼 **Confusion Matrix PNGs**  
   - 🖼 **ROC Curve PNGs**  

#### Generated Outputs
- `xgboost_confusion_matrix.png`  
- `xgboost_roc_curve.png`  
- `neural_net_confusion_matrix.png`  
- `neural_net_roc_curve.png`  

The script also prints a **side-by-side comparison table** of test metrics.

---

### 4️⃣ **utils.py**
Contains shared helper utilities used across the entire project.  
📄 Source: :contentReference[oaicite:3]{index=3}

#### Key Features
- Global path constants:
  - `DATA_PATH = data/diabetes_prediction_dataset.csv`
  - `RESULTS_DIR = results/`  
- Directory creation (`ensure_directories()`)  
- Random seed control for reproducibility (`seed_everything()`)  
- Optional PyTorch seeding support  
- Keeps script logic clean and centralized  

---

## 🧠 Overall Pipeline Flow

                 ┌─────────────────────────┐
                 │ diabetes_prediction.csv │
                 └──────────────┬──────────┘
                                │
                                ▼
                    preprocessing.py
                ┌───────────────────────────┐
                │ load_raw_data()           │
                │ train_val_test_split()    │
                │ fit_transform_preprocessor│
                └──────────────┬────────────┘
                               │
                               ▼
                         train.py
                ┌───────────────────────────┐
                │ Train XGBoost             │
                │ Train Neural Net          │
                │ Compute validation metrics│
                │ Save models + preprocessor│
                └──────────────┬────────────┘
                               │
                               ▼
                        evaluate.py
           ┌────────────────────────────────────────┐
           │ Load saved models + preprocessor       │
           │ Recreate train/val/test split          │
           │ Transform X_test                       │
           │ Evaluate XGBoost + NN                  │
           │ Save Confusion Matrices                │
           │ Save ROC Curves                        │
           └────────────────────────────────────────┘

---

## ✔ How This Folder Meets Project Requirements

🍏 **Full preprocessing pipeline**  
🍏 **Two best-performing models (XGBoost + NN)**  
🍏 **Proper model saving for reproducibility**  
🍏 **Evaluation on held-out test data**  
🍏 **Generated visualizations for the results folder**  
🍏 **Professional, modular, clean Python design**  

Everything required by your FAU AI project is implemented exactly as expected.

---

## 📌 Summary

The `src/` folder is the core engine of the Diabetes Prediction AI system. It contains:

- All preprocessing logic  
- Model training logic for both selected algorithms  
- Evaluation logic + visualization outputs  
- Utility infrastructure for reproducibility and directory control  

These scripts together provide a complete and reproducible ML pipeline from data → model → evaluation.

