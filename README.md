# 👥 Employee Attrition Predictor

Predicts whether an employee is at risk of leaving, based on their profile and work conditions, using a Gradient Boosting Classifier.

Built as part of a hands-on machine learning mini-project series.

---

## 🔍 Overview

Enter an employee's age, monthly income, years at the company, job satisfaction, and overtime status, and the model predicts whether they're likely to stay or likely to leave.

---

## 📊 Dataset

- 400 synthetic employee records
- Features: `Age`, `MonthlyIncome`, `YearsAtCompany`, `JobSatisfaction`, `OverTime`
- Target: `Attrition` (Yes / No)
- Class split: ~63% stayed, ~37% left — imbalanced, like most real attrition data

---

## 🤖 Model

```python
GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
```

**Test performance:** 73.8% accuracy — modestly above the ~71% "always predict stay" baseline. More importantly, recall on the attrition class is 65%, meaning it correctly flags about two-thirds of employees who actually leave. Accuracy alone would have hidden that class imbalance.

---

## 🛠️ Tech Stack

Python · pandas · scikit-learn

---

## 🚀 Getting Started

### Run the script

```bash
pip install pandas scikit-learn
python employee_attrition_prediction.py
```

### Run the web app

```bash
pip install streamlit pandas scikit-learn matplotlib
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`. Enter an employee profile, hit **Predict Attrition Risk**, and see the result alongside a chart of what actually drives the model's decisions.

**[ Screenshot pending — insert a screenshot of the running app here ]**

---

## ⚠️ Limitations

- Trained on synthetic data, not real HR records
- Only modestly beats the majority-class baseline — treat this as a demonstration of the approach, not a production-ready retention tool
- Doesn't account for factors like manager relationships, career growth opportunities, or company-wide events (layoffs, restructuring)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
