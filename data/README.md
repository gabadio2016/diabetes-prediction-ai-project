# Data Folder — Diabetes Prediction AI Project

This folder contains the dataset used for training, validating, and evaluating all machine-learning models in the Diabetes Prediction AI Project. The dataset provides health-related behavioral and biometric indicators that help classify whether an individual is at risk for diabetes.

---

##  File Included

### **`diabetes_prediction_dataset.csv`**
This is the primary dataset for the entire project. It contains thousands of health-indicator records, each labeled with a binary outcome for diabetes risk.

The dataset is used for:

- Exploratory data analysis (EDA)
- Feature engineering
- Model training (XGBoost & Neural Network)
- Model evaluation
- Performance comparisons between algorithms

---

## 📊 Dataset Structure

Although fields may vary slightly by version, the dataset generally contains the following variable types:

### **Health Metrics**
- BMI  
- PhysicalActivity  
- SleepTime  
- BloodPressure-related features (if provided)  

### **Lifestyle & Behavioral Indicators**
- Smoking  
- AlcoholDrinking  
- Stroke  
- DiffWalking  
- MentalHealth  
- PhysicalHealth  

### **Demographic Indicators**
- AgeCategory  
- Gender  
- Race  
- Diabetic (target variable)

### **Target Column**
- **Diabetic** — Binary classification label  
  - `1` → Indicates diabetes or high likelihood  
  - `0` → No diabetes indication

---

##  Preprocessing Notes

The dataset is prepared by the script **`src/preprocessing.py`**, which performs:

- Missing value handling  
- Encoding categorical variables  
- Converting string categories (e.g., “Yes/No”) into numerical values  
- Splitting the dataset into **training** and **testing** sets  
- Scaling/normalization when needed  

These processed outputs are then fed into the training scripts.

---

##  How This Folder Is Used in the Project

| Step | Script | How Data Is Used |
|------|--------|------------------|
| ✔ Load dataset | `preprocessing.py` | Reads the CSV file into a DataFrame |
| ✔ Clean + encode | `preprocessing.py` | Applies transformations to prepare features |
| ✔ Train models | `training.py` | Uses cleaned X_train / y_train |
| ✔ Evaluate models | `evaluation.py` | Loads X_test / y_test + saved models |
| ✔ Store results | `results/` | Outputs confusion matrices, ROC curves, metrics |

---

##  Best Practices

- **Do not modify the uploaded CSV.**  
  Always preserve the original dataset.

- If you create transformed versions (encoded, normalized), save them in a new folder such as:
