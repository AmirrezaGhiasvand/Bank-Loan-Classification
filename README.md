# Bank Personal Loan Prediction
Predicting **whether a bank customer will accept a personal loan offer** using demographic and financial data, with a focus on handling class imbalance and proper preprocessing.

## What This Project Does
Most introductory ML projects use balanced datasets and skip preprocessing decisions — this project tackles a real-world imbalanced classification problem and shows **why those decisions matter**.

Three classifiers are built and compared:
- **Logistic Regression** (with `StandardScaler` + `class_weight='balanced'`)
- **K-Nearest Neighbors** (with `StandardScaler` + automatic K selection)
- **Complement Naive Bayes**

## Key Concepts Covered
- **Class imbalance** — only ~9% of customers accepted the loan; accuracy alone is misleading, so F1 Score and Recall are the primary metrics
- **Feature scaling** — why `StandardScaler` is essential for distance-based (KNN) and gradient-based (Logistic Regression) models
- **Pipeline** — fitting the scaler inside a `Pipeline` to prevent data leakage from the test set
- **`class_weight='balanced'`** — penalising minority-class misses to improve recall on actual loan acceptors
- **ZIP code noise detection** — identifying and removing geographic outliers, then enriching the dataset with county and coordinates
- **K selection** — sweeping K = 1…29 and picking the elbow automatically rather than hardcoding

## Results

Metrics below are for the minority class (Loan = 1), which is what matters in an imbalanced setting.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.90 | 0.48 | 0.92 | 0.63 |
| KNN | 0.96 | 0.86 | 0.76 | 0.81 |
| Naive Bayes | 0.76 | 0.26 | 0.80 | 0.39 |

**KNN** achieves the best accuracy and F1 — it learns the decision boundary well once features are properly scaled. **Logistic Regression** trades precision for recall (0.92 recall on loan acceptors), which is arguably the most useful behaviour for a bank — it catches 92% of customers who would say yes, at the cost of some false positives. **Naive Bayes** gets high recall but collapses in precision (0.26), making it unreliable in practice.

## Tech Stack
- Python, Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- zipcodes

## Project Structure
```
├── data/
│   └── Bank_Personal_Loan_Modelling.csv
├── images/
├── src/
│   └── bankproject.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook src/bankproject.ipynb
```

The dataset should be placed at `data/Bank_Personal_Loan_Modelling.csv`. It is available on [Kaggle](https://www.kaggle.com/datasets/teertha/personal-loan-modelling).