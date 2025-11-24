# Notebooks Folder — Diabetes Prediction AI Project

This folder contains all exploratory, analytical, and experimental Jupyter Notebooks used throughout the Diabetes Prediction AI Project. Each notebook focuses on a specific stage of the machine-learning workflow: data quality verification, feature analysis, baseline testing, and model selection.

These notebooks were used for development, early experimentation, visualizations, and comparing candidate models before finalizing the training pipeline in the `src/` directory.

---

##  Notebook Overview

###  **data_quality_report.ipynb**
**Purpose:**  
Ensures that the dataset is clean, consistent, and ready for preprocessing and modeling.

**Main Tasks Performed:**
- Inspection of missing values  
- Summary statistics of all features  
- Detection of duplicates  
- Validity checks on ranges and categorical values  
- Visual distribution checks (histograms, counts)  
- Basic correlation heatmaps to detect multicollinearity

**Output:**  
A formal assessment confirming whether the dataset can be used for feature engineering and modeling.

---

###  **feature_analysis.ipynb**
**Purpose:**  
Examines the predictive power and behavior of each feature.

**Main Tasks Performed:**
- Data visualization of key variables  
- Feature importance estimation (statistical relationships or model-based)  
- Encoding checks for categorical variables  
- Correlation with the target variable  
- Removing redundant, irrelevant, or weak predictors  
- Understanding what features will matter most for ML

**Output:**  
A set of recommended features to use in model selection and training.

---

###  **Model_Evaluation_and_Baseline_Testing_updated.ipynb**
**Purpose:**  
Runs preliminary experiments to identify baseline performance before full model development.

**Main Tasks Performed:**
- Splitting data into train/test sets  
- Training simple baseline models  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Naive Bayes  
- Evaluating metrics: Accuracy, Precision, Recall, F1-Score  
- Visualizing confusion matrices and ROC curves  
- Documenting strengths/weaknesses of each baseline model  

**Output:**  
A benchmark comparison to guide which models deserve deeper optimization.

---

###  **model_selection.ipynb**
**Purpose:**  
Tests multiple models to determine which ones perform best for final training.

**Main Tasks Performed:**
- Implementing several ML models  
- Hyperparameter trials  
- Cross-validation testing  
- Ranking models by AUC, accuracy, and F1-Score  
- Selecting two best models (in your case: **XGBoost** and **Neural Network**)  
- Preparing notes for the final training script inside the `src/` folder  

**Output:**  
A justified selection of the top-performing models used in `training.py`.

---

##  Workflow Order

Recommended order for using the notebooks:

1. **data_quality_report.ipynb**  
2. **feature_analysis.ipynb**  
3. **Model_Evaluation_and_Baseline_Testing_updated.ipynb**  
4. **model_selection.ipynb**

Each notebook builds on the previous one, moving from raw data → insights → baseline tests → final model decisions.

---

##  Notes for Reproducibility

- Notebooks should run in a Python environment with all necessary libraries installed (pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost, TensorFlow).  
- Ensure the dataset path matches the structure inside the `data/` folder.  
- Export important plots into the `results/` folder for documentation and presentation.  
- The final ML pipeline is implemented in `/src`, while notebooks are mainly for experimentation.

---

##  Summary

This folder documents the exploratory analysis and decision-making steps that shaped the final modeling pipeline. Each notebook serves a specific purpose: verifying data quality, analyzing features, establishing baselines, and selecting the final two models (XGBoost + Neural Network) for full implementation.

