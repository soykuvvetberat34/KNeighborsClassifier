import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\sınıflandırma\\diabetes.csv")
y=df["Outcome"]
df=df.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.25,random_state=200)

knn_model=KNeighborsClassifier()
knn_params={
    "n_neighbors": np.arange(1,50)
}
knn_cv=GridSearchCV(knn_model,knn_params,cv=10,n_jobs=-1,verbose=2)
knn_cv.fit(x_train,y_train)
best_score=knn_cv.best_score_
n_neighbors=knn_cv.best_params_["n_neighbors"]
knn_tuned=KNeighborsClassifier(n_neighbors=n_neighbors)
knn_tuned.fit(x_train,y_train)
pred=knn_tuned.predict(x_test)
accuracy_score_=accuracy_score(y_test,pred)
print(accuracy_score_)
#accuracy score u yukaradki gibi bulabilirsin veya predict yapmadan direk
accuracy_score_2=knn_tuned.score(x_test,y_test)
print(accuracy_score_2)
#iki accuracy score da aynı sonucu döndürür













