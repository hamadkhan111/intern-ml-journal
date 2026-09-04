import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


df = sns.load_dataset("titanic")

df.head()
df.shape
df.columns
df.info()
df.isnull().sum()
df["survived"].value_counts()
df["survived"].value_counts(normalize=True)
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

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)


logistic_accuracy=accuracy_score(y_test,logistic_pred)
logistic_precision=precision_score(y_test,logistic_pred)
logistic_recall=recall_score(y_test,logistic_pred)
logistic_f1=f1_score(y_test,logistic_pred)

print("logistic Regression")
print("accuracy",logistic_accuracy)
print("precision",logistic_precision)
print("recall",logistic_recall)
print("recall",logistic_recall)
print("F1 score",logistic_f1)

decision_tree=Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",DecisionTreeClassifier(
        random_state=42
    ))
])
decision_tree.fit(X_train,y_train)
dt_pred=decision_tree.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

print("Decision Tree")
print("Accuracy :", dt_accuracy)
print("Precision:", dt_precision)
print("Recall   :", dt_recall)
print("F1 Score :", dt_f1)

random_forest=Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier(
    n_estimators=100,
    random_state=42
    ))
])

random_forest.fit(X_train,y_train)
rf_pred=random_forest.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("Random Forest")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)


gradient_bosting=Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",GradientBoostingClassifier(
        random_state=42
    ))
])


gradient_bosting.fit(X_train,y_train)
gb_pred=gradient_bosting.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)

print("Gradient Boosting")
print("Accuracy :", gb_accuracy)
print("Precision:", gb_precision)
print("Recall   :", gb_recall)
print("F1 Score :", gb_f1)


results=pd.DataFrame({
    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy":[
        logistic_accuracy,
        dt_accuracy,
        rf_accuracy,
        gb_accuracy
    ],
    "Precision":[
        logistic_precision,
        dt_precision,
        rf_precision,
        gb_precision
    ],
    "Recall":
    [
        logistic_recall,
        dt_recall,
        rf_recall,
        gb_recall
        ],
    "F1 Score":[
        logistic_f1,
        dt_f1,
        rf_f1,
        gb_f1
    ]
})


print(results)

ax = results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Comparison of Classification Models")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.legend(title="Metrics")
plt.tight_layout()

plt.show()
