# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
```
 Step 1. Start
 Step 2. Load the California Housing dataset.
 Step 3. select the first 3 features as input (X) and target variables (Y).
 Step 4. Split the data into training and testing sets .
 Step 5. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on 
 Step 6. Make predictions on the test data, inverse transform the predictions.
 Step 7. Then Calculate the Mean Squared Error.
 Step 8. End
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KAVIYA SNEKA M
RegisterNumber:  212223040091
/*
import pandas as pd
data=pd.read_csv("C:/Users/black/Downloads/Placement_Data.csv")
data.head()
 data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

![image](https://github.com/user-attachments/assets/8f3d932e-ac69-40fc-bd0a-8eb51f71747d)


![image](https://github.com/user-attachments/assets/c587504c-0608-4224-8341-ac9c798633da)


![image](https://github.com/user-attachments/assets/0bd06e3c-6de2-432b-a9ca-f18614ae7779)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
