import numpy as np
import pandas as pd
indata=pd.read_csv('train.csv')
y=indata.pop('Survived')
numeric_values=list(indata.dtypes[indata.dtypes!='object'].index)
x=indata[numeric_values]
x['Age'].fillna(x.Age.mean(),inplace=True)
from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier(n_estimators=200)
random.fit(x,y)
from sklearn.metrics import accuracy_score
print('accuracy for training set',accuracy_score(y,random.predict(x)))

#-----for test set---------------
test=pd.read_csv('test.csv')
test['Age'].fillna(test.Age.mean(),inplace=True)
test=test[numeric_values].fillna(test.mean()).copy()
y_predict=random.predict(test)
Submission=pd.DataFrame({
        'PassengerId':test['PassengerId'] ,'Survived':y_predict        
        })
Submission.to_csv('Titanic.csv',index=False)
