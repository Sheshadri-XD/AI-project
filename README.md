!pip install -q kagglehub

import os, kagglehub
import pandas as pd
import numpy as np

path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")
print("Files:", os.listdir(path))

df = pd.read_csv(os.path.join(path, "survey.csv"))
print(df.shape)
df.head()

df.info()
print(df.isnull().sum())
print(df["treatment"].value_counts())
# drop columns that aren't useful
df = df.drop(columns=["Timestamp", "comments", "state"], errors="ignore")

# 1) Age has impossible values (negative, 99999999 etc.)
df = df[(df["Age"] >= 18) & (df["Age"] <= 75)]

# 2) Gender is free text with dozens of spellings -> 3 groups
def clean_gender(g):
    g = str(g).strip().lower()
    if g in ["male", "m", "man", "cis male", "cis man", "male-ish", "maile",
             "mal", "male (cis)", "make", "guy (-ish) ^_^", "msle", "mail",
             "malr", "cis-male"]:
        return "Male"
    if g in ["female", "f", "woman", "cis female", "cis-female/femme",
             "female (cis)", "femake", "femail"]:
        return "Female"
    return "Other"

df["Gender"] = df["Gender"].apply(clean_gender)

# 3) Missing values
df["self_employed"] = df["self_employed"].fillna(df["self_employed"].mode()[0])
df["work_interfere"] = df["work_interfere"].fillna("Don't know")

print(df.isnull().sum().sum(), "missing values left")
print(df["Gender"].value_counts())
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
sns.countplot(x="treatment", data=df, ax=ax[0])
sns.countplot(x="family_history", hue="treatment", data=df, ax=ax[1])
sns.countplot(x="work_interfere", hue="treatment", data=df, ax=ax[2])
plt.tight_layout(); plt.show()
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

y = df["treatment"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["treatment"])

num_cols = ["Age"]
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Pipeline([("pre", pre),
                  ("clf", LogisticRegression(max_iter=2000))])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

pred = model.predict(X_test)
print("Test Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, target_names=["No treatment", "Treatment"]))

cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("5-fold CV accuracy:", cv.mean().round(3))

sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.show()
names = model.named_steps["pre"].get_feature_names_out()
coef = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()

print("Top factors increasing likelihood of treatment:\n", coef.tail(8))
coef.tail(10).plot(kind="barh", title="Top factors linked to seeking treatment")
plt.show()
sample = X_test.iloc[[0]]
prob = model.predict_proba(sample)[0][1]
print(f"Estimated risk probability: {prob:.2f}")
import os
from google.colab import files
os.makedirs("static", exist_ok=True)
files.upload()
os.replace("index.html", "static/index.html")
import threading
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from google.colab import output

app = Flask(__name__, static_folder="static")

FIELDS = ["Age", "Gender", "Country", "self_employed", "family_history",
          "work_interfere", "no_employees", "remote_work", "tech_company",
          "benefits", "care_options", "wellness_program", "seek_help",
          "anonymity", "leave", "mental_health_consequence",
          "phys_health_consequence", "coworkers", "supervisor",
          "mental_health_interview", "phys_health_interview",
          "mental_vs_physical", "obs_consequence"]

@app.get("/")
def home():
    return send_from_directory("static", "index.html")

@app.post("/api/predict")
def predict():
    d = request.get_json(silent=True) or {}
    missing = [f for f in FIELDS if d.get(f) in (None, "")]
    if missing:
        return jsonify(error="Missing: " + ", ".join(missing)), 400
    row = {f: d[f] for f in FIELDS}
    row["Age"] = int(row["Age"])
    row["Gender"] = clean_gender(row["Gender"])
    prob = float(model.predict_proba(pd.DataFrame([row]))[0][1])
    band = "Low" if prob < 0.35 else "Moderate" if prob < 0.65 else "High"
    return jsonify(probability=round(prob, 3), band=band)

threading.Thread(
    target=lambda: app.run(port=5000, debug=False, use_reloader=False),
    daemon=True).start()

output.serve_kernel_port_as_iframe(5000, height=900)
