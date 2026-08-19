import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
# Expected columns: Age, MonthlyIncome, YearsAtCompany, JobSatisfaction, OverTime, Attrition
data = pd.read_csv("employee_attrition_data.csv")

# Convert Categorical Data
le_overtime = LabelEncoder()
le_attrition = LabelEncoder()

data["OverTime"] = le_overtime.fit_transform(data["OverTime"])
data["Attrition"] = le_attrition.fit_transform(data["Attrition"])

# Features and Target
X = data[["Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction", "OverTime"]]
y = data["Attrition"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Test Accuracy
prediction = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, prediction))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))
print("\nClassification Report")
print(classification_report(y_test, prediction))

# User Prediction
age = float(input("Enter Age: "))
monthly_income = float(input("Enter Monthly Income ($): "))
years_at_company = float(input("Enter Years at Company: "))
job_satisfaction = float(input("Enter Job Satisfaction (1=Low, 2=Medium, 3=High, 4=Very High): "))
overtime = input("Works Overtime? (Yes/No): ")

overtime_enc = le_overtime.transform([overtime])[0]

new_employee = [[age, monthly_income, years_at_company, job_satisfaction, overtime_enc]]

prediction = model.predict(new_employee)

if prediction[0] == 1:
    print("\nPrediction: Likely to Leave (Attrition Risk)")
else:
    print("\nPrediction: Likely to Stay")
