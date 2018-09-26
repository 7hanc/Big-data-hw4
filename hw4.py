
import pandas as pd
import numpy as np
_2007=pd.read_csv("./Documents/2007.csv")


#choose month and day_of_month as features
month7=_2007["Month"]
dm7=_2007["DayofMonth"]
#weather_delay is the result that we want to predict (called predictors)
w_delay7=_2007["WeatherDelay"]
s=(len(_2007),2)
#m: store the features
#w: store the predictors
m=np.zeros(s,dtype=np.int) 
w=np.zeros((len(_2007),1))  


for i in range(len(_2007)):
    m[i][0]=month7[i]
    m[i][1]=dm7[i]
    w[i]=w_delay7[i]

#cross validation
#use kfold to validate my model(and will get the index of training data and testing data)
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
kfold_train = []
kfold_test=[]
for train_index, test_index in kf.split(m):
    kfold_train.append(train_index)
    kfold_test.append(test_index)


#use linear regression to predict
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
#use kfold[0] to build the model_1 and predict(x_train, y_train, x_test, y_test)
s2=(len(kfold_train[0]),2)
s3=(len(kfold_test[0]),2)
x_train1=np.zeros(s2,dtype=np.int)
y_train1=np.zeros((len(kfold_train[0]),1))
x_test1=np.zeros(s3,dtype=np.int)
y_test1=np.zeros((len(kfold_test[0]),1))

for i in range(len(kfold_train[0])):
    train_index=kfold_train[0][i]
    x_train[i]=m[train_index]
    y_train[i]=w[train_index]
    
for i in range(len(kfold_test[0])):
    test_index=kfold_test[0][i]
    x_test1[i]=m[test_index]
    y_test1[i]=w[test_index]
    
lm.fit(x_train1, y_train1)
y_test_predicted1=lm.predict(x_test1)



#use kfold[1] to build the model_2 and predict(x_train, y_train, x_test, y_test)
s4=(len(kfold_train[1]),2)
s5=(len(kfold_test[1]),2)
x_train2=np.zeros(s4,dtype=np.int)
y_train2=np.zeros((len(kfold_train[1]),1))
x_test2=np.zeros(s5,dtype=np.int)
y_test2=np.zeros((len(kfold_test[1]),1))

for i in range(len(kfold_train[1])):
    train_index=kfold_train[1][i]
    x_train2[i]=m[train_index]
    y_train2[i]=w[train_index]
    
for i in range(len(kfold_test[1])):
    test_index=kfold_test[1][i]
    x_test2[i]=m[test_index]
    y_test2[i]=w[test_index]

lm.fit(x_train2, y_train2)
y_test_predicted2=lm.predict(x_test2)



#mae of cross validation (k1)
mae_diff1=np.zeros((len(y_test1),1))
for i in range(len(y_test1)):
    mae_diff1[i]=y_test1[i] - y_test_predicted1[i]
mae1 = np.mean(np.absolute( mae_diff1 ))

#mae of cross validation (k2)
mae_diff2=np.zeros((len(y_test2),1))
for i in range(len(y_test2)):
    mae_diff2[i]=y_test2[i] - y_test_predicted2[i]
mae2 = np.mean(np.absolute( mae_diff2 ))
cv_mae=(mae1+mae2)/2



#rmse of cross validation (k1)
rmse_diff1=np.zeros((len(y_test1),1))
for i in range(len(y_test1)):
    rmse_diff1[i]=y_test1[i] - y_test_predicted1[i]
rmse1 = np.sqrt(np.mean(pow(rmse_diff1,2)))

#rmse of cross validation (k2)
rmse_diff2=np.zeros((len(y_test2),1))
for i in range(len(y_test2)):
    y_diff2[i]=y_test2[i] - y_test_predicted2[i]
rmse2 = np.sqrt(np.mean(pow(rmse_diff2,2)))
cv_rmse=(rmse1+rmse2)/2


#load the data as testing
_2008=pd.read_csv("./Documents/2008.csv")
w_delay8=_2008["WeatherDelay"]


#because the origin of w_delay8 has the nan, we need to change nan to zero 
w_delay8=np.nan_to_num(w_delay8)


#mae of the standard data(2008)
mae_diff_true=np.zeros((len(y_test_predicted1),1))
for i in range(len(y_test_predicted1)):
    mae_diff_true[i]=w_delay8[i] - y_test_predicted1[i]
mae_true = np.mean(np.absolute( mae_diff_true ))


#rmse of the standard data(2008)
rmse_diff_true=np.zeros((len(y_test_predicted1),1))
for i in range(len(y_test1)):
    rmse_diff_true[i]=w_delay8[i] - y_test_predicted1[i]
rmse_true = np.sqrt(np.mean(pow(rmse_diff_true,2)))

print(cv_mae)       #mae of validation in training 
print(cv_rmse)      #rmse of validation in training
print(mae_true)     #mae of prediction in testing
print(rmse_true)    #rmse of prediction in testing


