import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

df = sns.load_dataset("titanic")

print(df.head())
print(df.shape)
print(df.columns)
df.info()
print(df.isnull().sum())
print(df.describe())
print(df["survived"].value_counts())

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

categorical_features = ["sex", "embarked"]
numerical_features = ["pclass", "age", "sibsp", "parch", "fare"]

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

baseline_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(y_test, y_pred)
baseline_precision = precision_score(y_test, y_pred)
baseline_recall = recall_score(y_test, y_pred)
baseline_f1 = f1_score(y_test, y_pred)

print("BASELINE MODEL PERFORMANCE")
print("Accuracy:", baseline_accuracy)
print("Precision:", baseline_precision)
print("Recall:", baseline_recall)
print("F1 Score:", baseline_f1)

print(classification_report(y_test, y_pred))

df_fe = df.copy()

df_fe["FamilySize"] = df_fe["sibsp"] + df_fe["parch"] + 1
df_fe["IsAlone"] = (df_fe["FamilySize"] == 1).astype(int)
df_fe["FarePerPerson"] = df_fe["fare"] / df_fe["FamilySize"]

print(df_fe.head())

features_fe = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "FamilySize",
    "IsAlone",
    "FarePerPerson"
]

X_fe = df_fe[features_fe]
y_fe = df_fe["survived"]

X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_fe,
    y_fe,
    test_size=0.2,
    random_state=42,
    stratify=y_fe
)

numerical_features_fe = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "FamilySize",
    "IsAlone",
    "FarePerPerson"
]

categorical_features_fe = [
    "sex",
    "embarked"
]

numeric_pipeline_fe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline_fe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_fe = ColumnTransformer([
    ("num", numeric_pipeline_fe, numerical_features_fe),
    ("cat", categorical_pipeline_fe, categorical_features_fe)
])

feature_engineered_model = Pipeline([
    ("preprocessor", preprocessor_fe),
    ("classifier", LogisticRegression(max_iter=1000))
])

feature_engineered_model.fit(
    X_train_fe,
    y_train_fe
)

y_pred_fe = feature_engineered_model.predict(X_test_fe)

fe_accuracy = accuracy_score(y_test_fe, y_pred_fe)
fe_precision = precision_score(y_test_fe, y_pred_fe)
fe_recall = recall_score(y_test_fe, y_pred_fe)
fe_f1 = f1_score(y_test_fe, y_pred_fe)

print("FEATURE ENGINEERED MODEL")
print("Accuracy:", fe_accuracy)
print("Precision:", fe_precision)
print("Recall:", fe_recall)
print("F1 Score:", fe_f1)

print(classification_report(y_test_fe, y_pred_fe))

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Original Baseline": [
        baseline_accuracy,
        baseline_precision,
        baseline_recall,
        baseline_f1
    ],
    "Feature Engineered": [
        fe_accuracy,
        fe_precision,
        fe_recall,
        fe_f1
    ]
})

comparison["Improvement"] = (
    comparison["Feature Engineered"] -
    comparison["Original Baseline"]
)

print(comparison)

comparison_plot = comparison.set_index("Metric")[
    ["Original Baseline", "Feature Engineered"]
]

comparison_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Baseline vs Feature Engineering")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

print(df_fe["FamilySize"].value_counts().sort_index())
print(df_fe["IsAlone"].value_counts())

numeric_df = df_fe[
    [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "FamilySize",
        "IsAlone",
        "FarePerPerson"
    ]
]

correlation = numeric_df.corr()

print(
    correlation["survived"].sort_values(
        ascending=False
    )
)

plt.figure(figsize=(10, 7))

plt.imshow(
    correlation,
    aspect="auto"
)

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.colorbar()
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

model = feature_engineered_model.named_steps["classifier"]

feature_names = (
    feature_engineered_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = pd.Series(
    model.coef_[0],
    index=feature_names
)

print(
    coefficients
    .sort_values(
        key=abs,
        ascending=False
    )
    .head(15)
)

top_features = coefficients.abs().sort_values(
    ascending=False
).head(10)

top_features.sort_values().plot(
    kind="barh",
    figsize=(10, 6)
)

plt.title("Top 10 Features by Absolute Logistic Regression Coefficient")
plt.xlabel("Absolute Coefficient")
plt.tight_layout()
plt.show()