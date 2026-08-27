import pandas as pd 
import numpy as np 
import seaborn as sns
from sympy import true
df=sns.load_dataset("titanic")
df.head()
print("rOW",df.shape[0])
print("column",df.shape[1])
print(df.columns.tolist())
df.info()
df.describe()
df.dtypes

print(df["sex"].value_counts())
print(df["sex"].value_counts(normalize=true)*100)
print("survival rate",df["survived"].mean()*100)

gender_survival=df.groupby("sex")["survived"].mean()*100
print(gender_survival)

passenger_survival=df.groupby("pclass")["survived"].mean()*100
print(passenger_survival)


passenger_sex_survival=df.groupby(["pclass","sex"])["survived"].mean()*100
print(passenger_sex_survival)


print(df["embarked"].value_counts())
averagefare=df.groupby("embarked")["fare"].mean()*100
print(averagefare)


missing_values=df.isna().sum()
print(missing_values)
percentmissing_VALUE=df.isna().mean()*100
print(percentmissing_VALUE)

clean_df=df.copy()
clean_df.isna().sum()
clean_df["age"]=clean_df["age"].fillna(clean_df["age"].median())

clean_df["embarked"]=clean_df["embarked"].fillna(clean_df["embarked"].mode()[0])

clean_df.isna().sum()

clean_df=clean_df.drop(columns=["deck"])
print(clean_df)

print(df.loc[0:4,["sex","age","survived"]])

print(df.iloc[0:5,0:5])

min50=df[df["age"] > 50]
print(min50)


print(df[df["age"] > 50])


print("Original shape:", df.shape)
print("Cleaned shape:", clean_df.shape)

print("\nRemaining missing values:")
print(clean_df.isna().sum())