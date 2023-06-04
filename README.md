# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required packages.

2.Read the data set.

3.Apply label encoder to the non-numerical column inoreder to convert into numerical values.

4.Determine training and test data set.

5.Apply decision tree regression on to the dataframe and get the values of Mean square error, r2 and data prediction.
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: G R PRASANNA
RegisterNumber:  212221040129
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

Data.head()

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128134963/e7c07ed6-69cf-4c4a-84ad-e27212690089)



Data.info()

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128134963/5a4e00c8-3c8d-473b-9d7d-bf75744ce21a)


Data.isnull().sum()


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128134963/d3aaadf1-d9d9-42c6-a70d-65211e7ab707)

Data.head() for salary

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128134963/5402ff14-9229-4eab-bbbf-dac446d2e2b0)


MSE value


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128134963/7e4e5e68-5ce3-4fd6-9d10-006f868ae306)


r2 value

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128134963/923b974c-50f1-4680-87df-418f065d1327)


Data prediction

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128134963/4de03f07-c0c8-4a17-aeec-291d8b9a28b2)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
