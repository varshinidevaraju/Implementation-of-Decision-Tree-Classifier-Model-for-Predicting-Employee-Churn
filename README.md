# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: D.VARSHINI 
RegisterNumber:  212223230234
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:
## Data Head:
![image](https://github.com/user-attachments/assets/4dfaef39-4b68-4af1-afc4-e36bd607afba)

## Dataset info :
![image](https://github.com/user-attachments/assets/e305227c-6ccb-4cbb-93cf-0ada1fe026a4)

## Null Dataset:
![image](https://github.com/user-attachments/assets/d5fe5c31-bee5-4275-8272-d7f2230e0dc0)

## Values count in left column:

![image](https://github.com/user-attachments/assets/fb1cbddd-03de-4160-9ec2-51555407a15c)

## Dataset transformed head:
![image](https://github.com/user-attachments/assets/ecabd9fd-584c-47c3-87ad-90d7c3e361b2)


## x.head:
![image](https://github.com/user-attachments/assets/476ae80c-7e9f-40eb-9b5d-551be6f984cb)


## Accuracy:
![image](https://github.com/user-attachments/assets/a5ab1109-9710-458c-90ea-315cd4bc4426)

## Data prediction:
![image](https://github.com/user-attachments/assets/299c1059-a14b-4c97-9920-35862d269e5c)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
