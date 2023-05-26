# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: REVATHI.D
RegisterNumber:  212221240045

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://github.com/Revathi-Dayalan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/96000574/d22bd7c5-16c7-4e90-8187-0799239ae9b3)

![image](https://github.com/Revathi-Dayalan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/96000574/d38ec25f-673a-4265-bc63-a18593a71cc0)

![image](https://github.com/Revathi-Dayalan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/96000574/d353d0c1-4993-4f71-a7c1-4963cde9ca1c)

![image](https://github.com/Revathi-Dayalan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/96000574/99082d50-3d76-41d2-ad49-00ebf853b065)

![image](https://github.com/Revathi-Dayalan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/96000574/5e4ad83e-03cc-4854-b367-b5c29517d97a)

![image](https://github.com/Revathi-Dayalan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/96000574/52c26fac-2d22-4843-838c-429c0ea3a7e0)

![image](https://github.com/Revathi-Dayalan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/96000574/c760eaeb-d4fe-47f4-aae9-265955da776e)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
