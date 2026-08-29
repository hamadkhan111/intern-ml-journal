import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



df=sns.load_dataset("titanic")
a=df.head()
print(a)

print("Row:",df.shape[0])
print("Column:",df.shape[1])

features=[
    "pclass",
    "sex", 
    "age",
    "sibsp",
    "parch",
    "fare", 
    "embarked"
]

target="survived"
X=df[features]
y=df[target]

print("features:",X.head())
print("Target values:",y.head())


numeric_features=["pclass","age","sibsp","parch","fare"]

categorical_features=["sex","embarked"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print("training set",X_train.shape[0])
print("testing set ",X_test.shape[0])


numeric_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])


categorical_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])


preprocessor=ColumnTransformer(
    transformers=[
        ("num",numeric_transformer,numeric_features),
        ("cat",categorical_transformer,categorical_features)
    ]
)

model=Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("classifier",LogisticRegression())
    ]
)

model.fit(X_train,y_train)


y_pred=model.predict(X_test)
print(y_pred[:20])


y_probability = model.predict_proba(X_test)

print(y_probability[:5])

acc=accuracy_score(y_test,y_pred)
print("Accuracy:",acc)

pres=precision_score(y_test,y_pred)
print("Precision:",pres)


recall=recall_score(y_test,y_pred)  
print("Recall:",recall)

f1=f1_score(y_test,y_pred)
print("F1 Score:",f1)

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",cm)

maj_class=y_train.mode()[0]
print("majority class",maj_class)

baseline_predictions=np.full(
    len(y_test),maj_class
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline Accuracy: {baseline_accuracy * 100:.2f}%")

comparison = pd.DataFrame({
    "Model": [
        "Majority Class Baseline",
        "Logistic Regression"
    ],
    "Accuracy": [
        baseline_accuracy,
        acc
    ]
})

comparison

improvement = acc - baseline_accuracy

print(f"Improvement over baseline: {improvement:.4f}")
print(f"Improvement: {improvement * 100:.2f} percentage points")