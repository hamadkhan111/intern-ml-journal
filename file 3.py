import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

df = sns.load_dataset("titanic")

print(df.head())

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)



plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("number of passenger")

plt.show()



plt.figure(figsize=(8,5))
sns.histplot(
    data=df,
    x="fare",
    bins=30,
    kde=True
)
plt.title("Distribution of Passenger Fare")
plt.xlabel("fare")
plt.ylabel("number of passengers")

plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="fare"
)

plt.title("Boxplot of Passenger Fare")
plt.xlabel("fare")
plt.show()



plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="sibsp",
    discrete=True
)

plt.title("Distribution of Siblings/Spouses Aboard")
plt.xlabel("Number of Siblings/Spouses")
plt.ylabel("Number of Passengers")

plt.show()



numeric_df = df.select_dtypes(include="number")

numeric_df.head()

correlation=numeric_df.corr()
correlation

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap of Titanic Numeric Features")

plt.show()