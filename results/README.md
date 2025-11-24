# Results Folder — Diabetes Prediction AI Project

This folder contains all saved model artifacts, preprocessing objects, and evaluation outputs generated during experimentation and final model training. These results document how well each selected model performs and provide visual evidence supporting the final model choices.

---

##  Files Included

###  Saved Model & Preprocessing Artifacts
These files are generated during training and are required for consistent prediction on new data.

| File | Description |
|------|-------------|
| `preprocessor.pkl` | Saved preprocessing pipeline (encoders, transformers). Ensures new data uses the exact same transformations as training data. |
| `scaler.pkl` | Feature scaler (e.g., StandardScaler or MinMaxScaler). Normalizes numeric features during training and inference. |
| `xgboost.pkl` | Final trained XGBoost model used for inference and evaluation. |
| `neural_network.pkl` | Final trained Neural Network model saved in serialized format. |

These objects allow full reproduction of results and can be used for deployment.

---

##  Evaluation Outputs (Images)

###  **XGBoost Evaluation**

#### **Confusion Matrix**
**File:** `xgboost_confusion_matrix.png`  
**Path:** `/mnt/data/xgboost_confusion_matrix.png`

Shows model predictions vs. true labels:
- True Negatives: **13,701**
- False Positives: **24**
- False Negatives: **393**
- True Positives: **882**

Interpretation:  
XGBoost is very strong at identifying non-diabetic cases, with extremely low false positives.

#### **ROC Curve**
**File:** `xgboost_roc_curve.png`  
**Path:** `/mnt/data/xgboost_roc_curve.png`

- **AUC: 0.979**  
- The model demonstrates excellent discriminative power.
- Curve hugs the upper-left corner, indicating high sensitivity and specificity.

---

###  **Neural Network Evaluation**

#### **Confusion Matrix**
**File:** `neural_net_confusion_matrix.png`  
**Path:** `/mnt/data/neural_net_confusion_matrix.png`

- True Negatives: **13,698**
- False Positives: **27**
- False Negatives: **407**
- True Positives: **868**

Interpretation:  
The neural network performs similarly to XGBoost, with slightly more false positives and false negatives.

#### **ROC Curve**
**File:** `neural_net_roc_curve.png`  
**Path:** `/mnt/data/neural_net_roc_curve.png`

- **AUC: 0.976**  
- Very close to XGBoost in performance, with strong overall classification ability.

---

##  Model Comparisons

| Model | AUC Score | Strengths |
|-------|-----------|-----------|
| **XGBoost** | **0.979** | Best overall performance, lowest false positives, great curve shape. |
| **Neural Network** | **0.976** | Strong performance, good generalization, slightly more errors. |

**Conclusion:**  
Both models perform extremely well, but **XGBoost edges out as the best model** due to:
- Higher AUC  
- Fewer false positives  
- More stable predictions  

---

##  Purpose of This Folder

This folder exists to:

- Store all saved model files for inference  
- Provide visual evaluation outputs for presentations and reports  
- Record baseline vs. final model performance  
- Ensure reproducible experimentation  

These results directly support the final model choices made in your project.

---

##  How These Files Are Used

| File | Used By | Purpose |
|------|---------|---------|
| `preprocessor.pkl` | `preprocessing.py`, deployment apps | Apply identical data transformations |
| `scaler.pkl` | `training.py`, `evaluation.py` | Ensure consistent normalized inputs |
| `xgboost.pkl` | `evaluation.py` | Load and evaluate XGBoost model |
| `neural_network.pkl` | `evaluation.py` | Load and evaluate Neural Net |
| Confusion matrix images | Presentation, README | Visual model comparison |
| ROC curve images | Presentation, README | AUC + predictive performance check |

---

##  Summary

This folder provides the **evidence**, **artifacts**, and **visual proof** that justify your final chosen models. It contains everything needed to:

- Compare model performance  
- Reproduce evaluation  
- Deploy the final models  
- Present findings clearly and professionally  

