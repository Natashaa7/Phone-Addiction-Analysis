# 📱 Phone Addiction Analysis


## 📌 Overview

This project analyzes smartphone addiction patterns in teenagers, identifying behavioral and demographic factors that contribute to addiction. The workflow involves data preprocessing, exploratory data analysis, feature selection, model comparison, and inference. The project uses multiple machine learning models to classify users into addiction risk categories and selects the best-performing model (Gradient Boosting).


## 🗂 Repository Structure 
```Phone-Addiction-Analysis/
│
├─ notebooks/
│   ├─ EDA.ipynb                  # Exploratory Data Analysis
│   ├─ Data-Preprocessing.ipynb   # Data cleaning and encoding
│   ├─ Feature-Selection.ipynb    # Feature importance and selection
│   ├─ Base-model.ipynb           # Baseline models
│   ├─ Comparison.ipynb           # Model performance comparison
│   ├─ Logistic-Regression.ipynb  # Logistic Regression model
│   ├─ Random-Forest.ipynb        # Random Forest model
│   ├─ Gradient-Boosting.ipynb    # Gradient Boosting model
│   ├─ XGBoost.ipynb              # XGBoost model
│   ├─ Best-Model(Gradient-Boosting).ipynb  # Retrain best model and save
│   ├─ Infererence-best-model.ipynb  # Predictions using saved model
│
├─ source/model/                   # Saved ML models using joblib
├─ teen_phone_addiction_dataset.csv
├─ [app.py](http://app.py/)                          # Web app or API entry point
├─ [inference.py](http://inference.py/)                    # Script for running inference
├─ requirements.txt
├─ .gitignore
├─ [README.md](http://readme.md/) ```


## 🧰 Tools & Libraries

Programming Language: Python

Data Analysis & Visualization: Pandas, NumPy, Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)

Serialization: Joblib (for saving models)

Notebook Interface: Jupyter Notebook

## 📝 Workflow & Notebooks

### 1. **EDA.ipynb**

- Performs **Exploratory Data Analysis**
- Visualizes distributions, correlations, and patterns in the dataset
- Identifies trends such as high phone usage before bedtime, app usage patterns, and their effect on stress or academic performance

### 2. **Data-Preprocessing.ipynb**

- Handles **missing values**
- Encodes categorical variables
- Normalizes/standardizes features if required
- Prepares data for modeling

### 3. **Feature-Selection.ipynb**

- Computes **feature importance** using multiple methods
- Reduces dimensionality by removing low-impact features
- Ensures that the final feature set improves model performance

### 4. **Base-model.ipynb**

- Trains **baseline models** to get a performance benchmark
- Simple initial implementations of classifiers

### 5. **Comparison.ipynb**

- Implements **four ensemble methods and Logistic Regression**:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
- Compares models using metrics such as accuracy, confusion matrix, and balanced accuracy
- Visualizes performance comparison

### 6. **Logistic-Regression.ipynb, Random-Forest.ipynb, Gradient-Boosting.ipynb, XGBoost.ipynb**

- Train and evaluate each model **individually**
- Include **line-by-line explanations**
- Output model metrics and visualizations (e.g., confusion matrix heatmaps)

### 7. **Best-Model(Gradient-Boosting).ipynb**

- Chooses **Gradient Boosting** as the best performing model
- Retrains it on the full training dataset
- Saves the trained model using **joblib** for future inference

### 8. **Infererence-best-model.ipynb**

- Loads the **saved Gradient Boosting model**
- Demonstrates how to make predictions on new/unseen data
- Explains each step of inference clearly

## 💾 Saving & Loading Models

- Models are saved in **`source/model`** using `joblib` for efficient serialization:

```python
import joblib

# Save model
joblib.dump(model, "source/model/gradient_boosting_model.joblib")

# Load model
model = joblib.load("source/model/gradient_boosting_model.joblib")

```

- Both **resampled datasets** (from SMOTE) and trained models are saved for reproducibility.

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/Natashaa7/Phone-Addiction-Analysis.git
cd Phone-Addiction-Analysis

```

1. Install dependencies:

```bash
pip install -r requirements.txt

```

1. Run notebooks in this order for full workflow:
    1. `EDA.ipynb` → `Data-Preprocessing.ipynb` → `Feature-Selection.ipynb` → `Base-model.ipynb` → `Comparison.ipynb` → model-specific notebooks
    2. `Best-Model(Gradient-Boosting).ipynb` → `Infererence-best-model.ipynb`
2. Alternatively, use `app.py` or `inference.py` for programmatic predictions.

## 🧠 What You’ll Learn

- 📊 Patterns in teen phone usage
- 🔍 Features that predict addiction level
- 🤖 Compare ML models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- 💾 Save & load trained models with **joblib**
- 🧪 How SMOTE helps with imbalanced data

## 🎯 Key Findings

- Teens using phones late at night have **higher stress levels**
- Frequent app switching indicates **higher addiction level**
- Gradient Boosting was the **best model** for prediction

## 📈 Visualizations

- Confusion matrix heatmaps for model evaluation
- Feature importance plots
- Screen time distributions and correlation heatmaps

![Workflow](docs/jupyter_workflow.gif) 
