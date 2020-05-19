import pandas as pd
import numpy
path_train="/home/goutham/kaggle/titanic/train.csv"
path_test="/home/goutham/kaggle/titanic/test.csv"
dt_train=pd.read_csv(path_train)
dt_test=pd.read_csv(path_test)
#print("*"*100)
#print(dt_train["Sex","Age"].head())
#print(dt_train.dtypes)
#dt_test["Sex"]=LabelBinarizer().fit_transform(dt_test["Sex"])
#print(dt_train.isnull().sum())
import seaborn as sns
import matplotlib.pyplot as py
sns.set()
#sns.scatterplot(x=dt_train["Cabin"],y=dt_train["Cabin"].count())
#py.show()
#sns.boxplot(x=dt_train["Age"])
#py.show()
#sns.boxplot(x=dt_train["Sex"],y=dt_train["Survived"])
#py.show()
#sns.lineplot(y=dt_train["Survived"],x=dt_train["Age"])
#py.show()
#sns.lineplot(y=dt_train["Survived"],x=dt_train["Pclass"])
#py.show()
#linear relation with negative slope between Pclass and Survived
#sns.scatterplot(x=dt_train["SibSp"],y=dt_train["Survived"])
#py.show()
#sns.lineplot(x=dt_train["Parch"],y=dt_train["Survived"])
#py.show()
#parameters that matter are: Pclass,Sex,Age,SibSp,Parch
#print(dt_train.dtypes)
################data processing#################################################################
dt_train=dt_train.fillna(dt_train.mean())
dt_test=dt_test.fillna(dt_test.mean())
dt_train["Age"]=dt_train["Age"].astype(int)
dt_test["Age"]=dt_test["Age"].astype(int)
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
dt_train["Sex"],dt_test["Sex"]=LabelBinarizer().fit_transform(dt_train["Sex"]),LabelBinarizer().fit_transform(dt_test["Sex"])
dt_train["Embarked"].fillna("S",inplace=True)
dt_train["Embarked"],dt_test["Embarked"]=LabelEncoder().fit_transform(dt_train["Embarked"]),LabelEncoder().fit_transform(dt_test["Embarked"])
dt_train=dt_train.drop(["Cabin","Ticket","Fare","Name"],axis=1)
dt_test=dt_test.drop(["Cabin","Ticket","Fare","Name"],axis=1)
x=dt_train.drop("Survived",axis=1)
y=dt_train.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=69)
#################################################################################################
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pipe=make_pipeline(StandardScaler(),SVC())
hyperparameters={'svc__C': [0.1, 1, 10, 100, 1000],'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],'svc__kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV
clf=GridSearchCV(pipe,hyperparameters,cv=10)
clf.fit(x_train,y_train)
y_predict=clf.predict(dt_test)
#from sklearn.metrics import classification_report
#print(classification_report(y_test,y_predict))
output = pd.DataFrame({'PassengerId': dt_test.PassengerId, 'Survived': y_predict})
output.to_csv("output.csv",index=False)