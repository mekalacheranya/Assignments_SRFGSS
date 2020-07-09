#KNN
import pandas as pd
import numpy as np
# assigning column_names
columns = ['Research','GRE_score','TOEFL_score','University_rating','SOP','LOR','cgpa','chance_of_Admit']
df = pd.read_csv("/storage/emulated/0/shaista/newfile.py",names=columns)
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.info())
# profile report
#profile = pp.ProfileReport(df)
#profile.to_file("/storage/emulated/0/shaista/newfileEDA.html")
from sklearn.model_selection import train_test_split
y = df['chance_of_Admit']
X = df.drop('chance_of_Admit',axis = 1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
# model building
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
#print(y_test)
#print(y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score
conf_mat = confusion_matrix(y_test,y_pred)
print(conf_mat)
acc_score = accuracy_score(y_test,y_pred)
print(acc_score)
# pre_deployment test
y_pred_new = knn.predict([[5.3,4,1.8,0.5]])
print(y_pred_new)