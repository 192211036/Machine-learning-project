import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --------------------------- Load and Preprocess Dataset ----------------------------

df = pd.read_csv(r"C:\Users\NITHISH\Desktop\Python projects\.venv\Employee attrition project(3)\Employee-Attrition - Employee-Attrition.csv")

# Drop unneeded columns
df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# One-hot encode categoricals
df = pd.get_dummies(df, drop_first=True)

# --------------------------- Feature Correlation ----------------------------

correlation_matrix = df.corr()
target_corr = correlation_matrix['Attrition'].abs().sort_values(ascending=False)
selected_features = target_corr[target_corr > 0.05].index.tolist()
selected_features.remove('Attrition')  # Remove target if present

# --------------------------- Train Multiple Models ----------------------------

X = df[selected_features]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

model_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_scores[name] = acc

# Save best model and feature columns
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")  # Save feature order

# --------------------------- Streamlit App UI ----------------------------

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("\U0001F6A8 Employee Attrition Prediction Dashboard")
st.markdown("### \U0001F468‍\U0001F4BC Analyze employee attrition trends and predict who's at risk")

# --------------------------- Home Page - View Data & Feature Selection ----------------------------
st.markdown("---")
st.markdown("## 🏠 Home Page: Dataset Preview and Feature Selection")

st.dataframe(df)

st.markdown("### 🎯 Select Features for Model Training")
feature_options = df.columns.tolist()
feature_options.remove("Attrition")

selected_input_features = st.multiselect("Choose features to use for prediction:", feature_options, default=selected_features)

# --------------------------- Sidebar Inputs ----------------------------

st.sidebar.header("\U0001F4CB Enter Employee Details for Prediction")

age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
distance = st.sidebar.slider("Distance from Home", 1, 30, 10)
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
work_life_balance = st.sidebar.slider("Work Life Balance (1-4)", 1, 4, 3)
training_times = st.sidebar.slider("Trainings Last Year", 0, 10, 2)
years_in_role = st.sidebar.slider("Years in Current Role", 0, 20, 3)
stock_option_level = st.sidebar.selectbox("Stock Option Level", [0, 1, 2, 3])

# Construct input dictionary with necessary fields
input_data = {
    "Age": age,
    "MonthlyIncome": income,
    "DistanceFromHome": distance,
    "JobSatisfaction": job_satisfaction,
    "YearsAtCompany": years_at_company,
    "WorkLifeBalance": work_life_balance,
    "TrainingTimesLastYear": training_times,
    "YearsInCurrentRole": years_in_role,
    "StockOptionLevel": stock_option_level,
    "OverTime_Yes": 1 if overtime == "Yes" else 0
}

# --------------------------- Prediction ----------------------------

model = joblib.load("best_model.pkl")
model_columns = joblib.load("model_columns.pkl")  # Load saved column order

# Create input DataFrame
input_df = pd.DataFrame([input_data])

# Add any missing columns with default 0
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match model's expected input
input_df = input_df[model_columns]

# Make prediction
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Show result
st.subheader("\U0001F4CA Prediction Result:")
if prediction == 1:
    st.error(f"\U0001F534 High Risk of Attrition (Probability: {probability:.2f})")
else:
    st.success(f"\U0001F7E2 Low Risk of Attrition (Probability: {probability:.2f})")

st.markdown("---")

# --------------------------- Tabs for Insights ----------------------------

tab1, tab2, tab3 = st.tabs(["\U0001F4C8 Correlation Matrix", "\U0001F4CA Model Comparison", "❓ HR Questions"])

with tab1:
    st.markdown("### \U0001F50D Correlation Matrix")
    corr = df.corr()
    plt.figure(figsize=(16, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    st.pyplot(plt)

with tab2:
    st.markdown("### \U0001F527 Model Accuracy Comparison")
    st.bar_chart(model_scores)
    st.markdown(f"✅ Best Model: **{best_model_name}** with Accuracy: **{model_scores[best_model_name]*100:.2f}%**")

with tab3:
    st.markdown("### \U0001F9E0 10 Important HR Questions")
    st.markdown("""
1. Does job satisfaction directly reduce attrition?  
2. Do employees working overtime tend to leave more?  
3. How does work-life balance affect retention?  
4. Is distance from home a key driver?  
5. Does income impact attrition?  
6. Do promotions reduce chances of leaving?  
7. Is low training associated with quitting?  
8. Does stock option level influence retention?  
9. Which department has the most attrition?  
10. What is the effect of performance rating?  
""")
