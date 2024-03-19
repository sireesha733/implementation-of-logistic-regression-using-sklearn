# implementation-of-logistic-regression-using-sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
dataset=pd.read_csv('/content/sample_data/Social_Network_Ads.csv')
print(dataset.columns)
x=dataset[['Age','EstimatedSalary']]
y=dataset['Purchased']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
#feature scaling
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print("Accuracy score",accuracy_score(y_test,y_pred))
#Model visualization:
sns.regplot(x=x_test[:,:-1],y=y_test,logistic=True)
