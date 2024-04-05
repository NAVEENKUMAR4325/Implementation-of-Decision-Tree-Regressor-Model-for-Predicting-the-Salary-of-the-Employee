# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the packages that helps to implement Decision Tree.
2. Download and upload required csv file or dataset for predecting Employee Churn
3. Initialize variables with required features.
4. And implement Decision tree classifier to predict Employee Churn

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Naveen Kumar E
RegisterNumber: 212222220029
*/
import pandas as pd
data=pd.read_csv('/content/Salary_EX7.csv')
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
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
plt.figure(figsize=(10,6))
plot_tree(dt,feature_names=x.columns,class_names=['Salary'], filled=True)
plt.show()
```

## Output:
### Dataset:

![image](https://github.com/shoaib3136/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117919362/200c1f04-279e-4c1c-9260-7796d07a6d26)

### Mean square error:

![image](https://github.com/shoaib3136/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117919362/426c563d-5c08-4d91-bcc4-3358c03bb85d)

### Testing of Model:

![image](https://github.com/shoaib3136/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117919362/b81d5534-b7ba-43ed-bae3-d1aea6d3216d)



### Decision Tree:

![image](https://github.com/shoaib3136/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117919362/3134c866-a852-4b49-96bf-0b29986b1993)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
