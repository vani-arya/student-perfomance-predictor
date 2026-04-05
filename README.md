# 🎓 Student Performance Predictor

A complete end-to-end Machine Learning project that predicts a student's **math exam score** based on demographic and academic features.

---

## 📁 Project Structure

```
student-performance/
│
├── data/                          # Place your dataset here (students.csv)
├── notebook/
│   └── eda.py                     # Exploratory Data Analysis
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Loads & splits data
│   │   ├── data_transformation.py # Preprocessing pipeline
│   │   └── model_trainer.py       # Trains & compares models
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py      # Orchestrates training
│   │   └── predict_pipeline.py    # Handles predictions
│   │
│   ├── exception.py               # Custom exception handler
│   ├── logger.py                  # Logging setup
│   └── utils.py                   # Helper functions
│
├── artifacts/                     # Auto-generated: model.pkl, preprocessor.pkl
├── logs/                          # Auto-generated log files
│
├── app.py                         # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone / download the project

```bash
cd student-performance
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Add your dataset

Download the **StudentsPerformance.csv** dataset from Kaggle:
https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

Rename it to `students.csv` and place it in the `data/` folder.

> If no dataset is provided, a synthetic dataset is auto-generated for demo purposes.

### 5. Run the training pipeline

```bash
python -m src.pipeline.train_pipeline
```

This will:
- Load and split the data
- Apply preprocessing (scaling + encoding)
- Train 4 ML models
- Save the best model to `artifacts/model.pkl`

### 6. Launch the Streamlit app

```bash
python -m streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🤖 Models Trained

| Model | Description |
|---|---|
| Linear Regression | Baseline model |
| Decision Tree | Interpretable tree model |
| Random Forest | Ensemble of trees |
| Gradient Boosting | Boosted ensemble (often best) |

The model with the highest **R² score** on the test set is automatically selected.

---

## 📊 Features Used

| Feature | Type |
|---|---|
| Gender | Categorical |
| Race / Ethnicity | Categorical |
| Parental Education Level | Categorical |
| Lunch Type | Categorical |
| Test Preparation Course | Categorical |
| Reading Score | Numerical |
| Writing Score | Numerical |

**Target:** Math Score (0–100)

---

## 📈 EDA

Run the EDA script to generate all plots:

```bash
python notebook/eda.py
```

Plots saved to `artifacts/`:
- Score distributions
- Correlation heatmap
- Scores by gender
- Test prep impact
- Parental education vs score
- Pairplot

---

## 🔧 Tech Stack

- **Python 3.10+**
- **scikit-learn** — ML models + pipelines
- **pandas / numpy** — Data handling
- **matplotlib / seaborn** — Visualizations
- **Streamlit** — Web app UI
- **pickle** — Model serialization

---

## 👨‍💻 Author

Built as a complete ML portfolio project.
