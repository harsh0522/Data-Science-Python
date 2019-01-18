# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix, auc ,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


os.chdir('')
# Importing the dataset
df=pd.read_csv('Social_Network_Ads.csv')

#Encoding
Gender = pd.get_dummies(df['Gender'],drop_first=True)
df.drop(['Gender'],axis=1,inplace=True)
df = pd.concat([df,Gender],axis=1)

df=df[['User ID','Age','EstimatedSalary','Male','Purchased']]





X = df.iloc[:, [1,2,3]].values #Removing User ID as it is not making any sense
y = df.iloc[:, 4].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X
X_train



logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
y_pred = logmodel.predict(X_test)


print(len(y_test),sum(y_test))

print(classification_report(y_test,y_pred))


print('accuracy %s' % accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix\n %s' % cm)

probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

auc(fpr, tpr)





