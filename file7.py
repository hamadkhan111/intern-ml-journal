import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error

housing=fetch_california_housing()

X=pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

y=pd.Series(
    housing.target,
    name="Target"
)


print("X_shape",X.shape)
print("y_shape",y.shape)

print(X.head)
print(X.info())
print(X.describe())
print(y.head())

X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)
print("training data",X_train.shape)
print("tetsing data",X_test.shape)

X_single=X[["MedInc"]]
X_train_single,X_test_single,y_train_single,y_test_single=train_test_split(
    X_single,y,
    test_size=0.2,
    random_state=42
)

degrees=[1,2,3,5,10,15]
train_errors=[]
test_errors=[]

for degree in degrees:
    model=Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("linear",LinearRegression())
    ])
    
    model.fit(X_train_single,y_train_single)
    
    train_prediction=model.predict(X_train_single)
    test_prediction=model.predict(X_test_single)
    
    train_error=mean_squared_error(
        y_train_single,
        train_prediction
    )
    
    test_error=mean_squared_error(
        y_test_single,
        test_prediction
    )
    train_errors.append(train_error)
    test_errors.append(test_error)
    
    
print("Train error:",train_error)
print("test error:",test_error)


plt.figure(figsize=(10,6))

plt.plot(
    degrees,
    train_errors,
    marker="o",
    label="training error"
)

plt.plot(
    degrees,
    test_errors,
    marker="o",
    label="testing error"
)

plt.xlabel("polynomial degree")
plt.ylabel("mean square error")

plt.title("train v test error polynomial")

plt.legend()
plt.grid(True)
plt.show()


alphas=[0.01,0.1,1,10,100]

ridge_train_errors=[]
ridge_test_errors=[]

for alpha in alphas:
    ridge_model=Pipeline([
        ("scaler",StandardScaler()),
        ("ridge",Ridge(alpha=alpha))
    ])
    
    
    ridge_model.fit(X_train,y_train)
    train_prediction=ridge_model.predict(X_train)
    test_prediction=ridge_model.predict(X_test)
    
    train_error=mean_squared_error(
        y_train,
        train_prediction
    )
    
    
    test_error=mean_squared_error(
        y_test,
        test_prediction
    )
    
    
    ridge_train_errors.append(train_error)
    ridge_test_errors.append(test_error)
    
    
print("RIdge train error",ridge_train_errors)
print("Ridge test error",ridge_test_errors)



plt.figure(figsize=(10,6))


plt.plot(
    alphas,
    ridge_train_errors,
    marker="o",
    label="Trainig error"
)


plt.plot(
    alphas,
    ridge_test_errors,
    marker="o",
    label="testing error"
)


plt.xscale("log")


plt.xlabel("Alpha")
plt.ylabel("Mean square error")

plt.title("ridge regresssion")


plt.legend()
plt.grid(True)

plt.show()



lasso_train_errors=[]
lasso_test_errors=[]

for alpha in alphas:
    lasso_model=Pipeline([
        ("scaler",StandardScaler()),
        ("lasso",Lasso(alpha=alpha,max_iter=10000))
    ])
    
    lasso_model.fit(X_train, y_train)
    
    
    train_prediction=lasso_model.predict(X_train)
    test_prediction=lasso_model.predict(X_test)
    
    train_error=mean_squared_error(
        y_train,
        train_prediction
    )
    
    test_error=mean_squared_error(
        y_test,
        test_prediction
    )
    
    lasso_train_errors.append(train_error)
    lasso_test_errors.append(test_error)
    
print("lasso train error",lasso_train_errors)
print("lasso test error",lasso_test_errors)


plt.figure(figsize=(10,6))

plt.plot(
    alphas,
    lasso_train_errors,
    marker="o",
    label="training data"
)


plt.plot(
    alphas,
    lasso_test_errors,
    marker="o",
    label="testing data"
)


plt.xscale("log")

plt.xlabel("Alpha")
plt.ylabel("Mean square error")

plt.title("lasso regression")

plt.legend()
plt.grid(True)
plt.show()



results=pd.DataFrame({
    "Alphas":alphas,
    "Ridge train error":ridge_train_errors,
    "Ridge test error":ridge_test_errors,    
    "Lasso train error":lasso_train_errors,
    "Lasso test error":lasso_test_errors
})

print(results)