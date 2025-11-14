# 🩺 Diabetes Prediction AI Project

### CAP 4630 – Intro to Artificial Intelligence  
**Team Members:** Gil Abadio, Sorahya Eugene, Tom Le, Sarai Aguiar, Erin Patrician


---

## 📘 Project Summary
This repository contains our AI project for predicting diabetes using health indicator data.  
We applied multiple **machine learning models**—including Logistic Regression, Random Forest, XGBoost, and a Neural Network—to classify whether an individual is at risk for diabetes based on survey and biometric inputs.

The project focuses on comparing models, evaluating accuracy, and understanding feature importance to interpret which health factors most influence diabetes prediction.

---

## 🧩 Folder Overview
diabetes-prediction-ai-project
│
├── data/ → Raw and processed dataset(s)
├── notebooks/ → Jupyter notebooks for analysis and model training
├── src/ → Python scripts for preprocessing and model code
├── results/ → Output metrics, confusion matrices, and graphs
├── presentation/ → Slides for final class presentation
│
├── LICENSE
└── README.md


---

## ⚙️ Setup Instructions
### Clone the repo
```bash
git clone https://github.com/gabadio2016/diabetes-prediction-ai-project.git
cd diabetes-prediction-ai-project
Install dependencies

pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
🧠 Workflow Summary
Data Preparation

Cleaned the dataset, removed duplicates, and normalized numerical columns.

Encoded categorical values (Yes/No → 1/0).

Model Training

Tested and compared several ML algorithms:

Logistic Regression

Random Forest

XGBoost

Neural Network (Keras/TensorFlow)

Evaluation

Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

Visual outputs in results/ folder (confusion matrices, ROC curves).

Presentation

Summary slides of our methodology and model results are in the presentation/ folder.

📊 Example Output
Model	Accuracy	AUC	Notes
Logistic Regression	~0.84	0.87	Fast, interpretable
Random Forest	~0.88	0.91	Good balance
XGBoost	~0.89	0.92	Best overall
Neural Network	~0.90	0.93	Slightly higher accuracy, longer training

(Exact metrics and graphs available in /results/.)

🧰 Tech Stack
Python 3.10+

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, XGBoost, TensorFlow

Tools: Jupyter Notebook, Google Colab

🚀 Next Steps
Tune hyperparameters for better model performance

Implement model explainability (SHAP values)

Deploy simple Streamlit or Flask interface for live predictions

🧾 Dataset
Based on CDC Diabetes Health Indicators dataset, including metrics such as BMI, blood pressure, cholesterol, smoking habits, and physical activity levels.
