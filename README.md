# 📱 Phone Addiction Analysis

## 📌 Overview

Ever wondered how phone usage affects stress, sleep, or school performance?

This project analyzes **teen smartphone addiction** patterns using data science and machine learning. We explore the data, test multiple models, and pick the best one to **predict addiction risk**. It analyzes smartphone addiction patterns in teenagers, identifying behavioral and demographic factors that contribute to addiction. The workflow involves **data preprocessing, exploratory data analysis, feature selection, model comparison, and inference**. The project uses multiple machine learning models to classify users into addiction risk categories and selects the best-performing model (Gradient Boosting).

## 📊 Dataset

This project uses the [**Teen Phone Addiction Dataset**](https://www.kaggle.com/datasets/sumedh1507/teen-phone-addiction?utm_source=chatgpt.com) from Kaggle.

### 📁 Dataset Overview:

- **Rows:** 300+ (teens surveyed)
- **Columns:** 15 (demographic, behavioral, and academic features)
- **Key Features:**
    - `Gender` – Male/Female
    - `Age` – Age of the teen
    - `Sleep Duration` – Average hours of sleep per night
    - `Study Hours` – Average study time per day
    - `Screen Time` – Daily smartphone usage in hours
    - `Social Media Usage` – Social media usage (hours)
    - `GPA` – Academic performance indicator
    - `Addiction Level` – Target label (Low, Medium, High)

### 📥 How to Get the Dataset:

1. Download from Kaggle: [Teen Phone Addiction Dataset]([https://www.kaggle.com/datasets/sumedh1507/teen-phone-addiction])
2. Place the file `teen_phone_addiction_dataset.csv` inside the **`notebooks/`** folder of this repository.
   
## 🗂 Repository Structure 
```python
📁 Phone-Addiction-Analysis
│
│── 📁 CLI_Inference/             # Command-line interface for predictions
│    └── [inference.py]             # Python script to run model inference via CLI
│
│── 📁 model/                     # Contains saved ML models
│    └── gb_tuned.joblib          # Final Gradient Boosting model (best performer)
│
│── 📁 notebooks/                 # Jupyter notebooks showing full workflow
│    ├── 1_EDA.ipynb              # Exploratory Data Analysis (visualizations, insights)
│    ├── 2_Data-Preprocessing.ipynb  # Data cleaning, encoding, scaling
│    ├── 3_Base-model.ipynb       # Logistic Regression baseline model
│    ├── 4_Feature-Selection.ipynb   # Feature importance & reduction
│    ├── 5_Random-Forest.ipynb    # Random Forest training & evaluation
│    ├── 6_XGBoost.ipynb          # XGBoost training & evaluation
│    ├── 7_Gradient-Boosting.ipynb # Gradient Boosting training
│    ├── 8_Logistic-Regression.ipynb # Logistic Regression detailed analysis
│    ├── 9_Comparison.ipynb       # Comparison of all models (metrics, graphs)
│    ├── 10_Best-Model(Gradient-Boosting).ipynb # Retraining tuned GB model
│    └── 11_Inference-best-model.ipynb # Inference notebook using final model
│
│── 📁 source/                    # FastAPI backend application
│    └── [app.py]                   # FastAPI app with SwaggerUI for API deployment
│
│── .gitignore                    # Ignore unnecessary files (envs, cache, etc.)
│── requirements.txt              # List of required Python dependencies
│── [README.md]                    # Project documentation (you are reading this)
```

## 🧰 Tools & Libraries

- **Programming Language:** Python
- **Data Analysis & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
- **Serialization:** Joblib (for saving models)
- **Notebook Interface:** Jupyter Notebook
- **Deployment:** FastAPI, Swagger UI

---

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

### 5. **Logistic-Regression.ipynb, Random-Forest.ipynb, Gradient-Boosting.ipynb, XGBoost.ipynb**

- Train and evaluate each model **individually**
- Include **line-by-line explanations**
- Output model metrics and visualizations (e.g., confusion matrix heatmaps)
  
### 6. **Comparison.ipynb**

- Implements **four ensemble methods and Logistic Regression**:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
- Compares models using metrics such as accuracy, confusion matrix, and balanced accuracy

### 7. **Best-Model(Gradient-Boosting).ipynb**

- Chooses **Gradient Boosting** as the best performing model
- Retrains it on the full training dataset
- Saves the trained model using **joblib** for future inference

### 8. **Infererence-best-model.ipynb**

- Loads the **saved Gradient Boosting model**
- Demonstrates how to make predictions on new/unseen data
- Explains each step of inference clearly

## 💾 Saving & Loading Models

- Models are saved in **`model`** using `joblib` for efficient serialization:

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

2. Create environment:
   
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

```

3. Install dependencies:

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

- Teens using phones late at night have **higher stress levels**.
- Frequent app switching indicates **higher addiction level**.
- Gradient Boosting was the **best model** for prediction.

## 📈 Visualizations

- Confusion matrix heatmaps for model evaluation
- Feature importance plots
- Screen time distributions and correlation heatmaps

## 💻 Deployment

- The best model is deployed using **FastAPI**
- **SwaggerUI** allows easy API interaction and testing
- Endpoints:
    - `/predict` – Make predictions for new data
    - `/docs` – Interactive API documentation via SwaggerUI

---

<img width="1278" height="819" alt="image" src="https://github.com/user-attachments/assets/da8c09d9-b463-4518-97e3-3fa6b105b03a" />

## 🎯 Key Learnings

- Handling imbalanced datasets with **SMOTE** improves model accuracy.
- Feature selection significantly impacts model performance.
- Ensemble methods like Gradient Boosting outperform simpler models in complex classification.
- FastAPI makes ML model deployment quick and scalable.
- SwaggerUI provides a user-friendly interface for testing APIs.

---

## 📝 Conclusion

This project successfully demonstrates:

- How smartphone usage can be analyzed to predict addiction risk
- The process of **data preprocessing, model training, comparison, and selection**
- Deployment of a trained model via **FastAPI** for real-time predictions

The **Gradient Boosting model** provides the best predictions and can be used to identify teens at risk of phone addiction.

---

## 🔮 Future Plans

- Utilize Docker to containerize this application, facilitating its convenient deployment.
- Integrate model deployment with a **front-end dashboard** for easier accessibility.
