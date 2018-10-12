from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
Location = r'C:\Users\your_pc\dataset_name.csv' #dataset name is train.csv
df = pd.read_csv(Location)
a=np.array(df)
y=a[:,55]
a=a[:,np.s_[0:55]]
def sigmoid(x):
    sig=1/(1+np.exp(-x))
    return sig
	
y1=np.zeros((7,15120))
for i in range(y.shape[0]):
    y1[y[i]-1,i]=1
    
            
print(y)
print(y1)
from sklearn import preprocessing
a = preprocessing.minmax_scale(a, feature_range=(-0.5, 0.5))
a_train=a[0:12000,:]
a_cvt=a[12000:15120,:]
y1_train=y1[:,np.s_[0:12000]]
y1_cvt=y1[:,np.s_[12000:15120]]
print(str(y1_train.shape)+" "+str(y1_cvt.shape))
w1=np.zeros((7,55))
b1=np.zeros((7,1))
a_train=a_train.T
a_cvt=a_cvt.T
def estimate(W,b,a):
    z=np.dot(W,a)+b
    yhat=sigmoid(z)
    return yhat

def cost(yhat,y):
    m=y.shape[1]
    cost1=-(np.sum(np.multiply(y,np.log(yhat))+np.multiply(1-y,np.log(1-yhat)),axis=1,keepdims=True)/m).T
    cost=np.sum(cost1,axis=1,keepdims=True)
    return cost

def gradient(yhat,a,y):
    m=y.shape[1]
    dw=np.dot(a,(yhat-y).T).T/m
    db=np.sum((yhat-y),axis=1,keepdims=True)/m
    grad={'dw':dw,'db':db}
    return grad

def optimize(num,learning_rate,W,b,a,y):
    for i in range(num):
        yhat=estimate(W,b,a)
        cost1=cost(yhat,y)
        grad=gradient(yhat,a,y)
        dw=grad['dw']
        db=grad['db']
        W=W-learning_rate*dw
        b=b-learning_rate*db
        if i %1000==0:
            print cost1
    parameter={'W':W,'b':b}
    return parameter

parameter=optimize(20000,0.12,w1,b1,a_train,y1_train)
def predict(parameter,a):
    w=parameter['W']
    b=parameter['b']
    z=np.dot(w,a)+b
    ypred=sigmoid(z)
    i=np.argmax(ypred, axis=0)
    ypred=np.zeros((7,a.shape[1]))
    for x in range(a.shape[1]):
        ypred[i[x],x]=1
    return ypred
	
y_pred=predict(parameter,a_cvt)
print(y_pred)
x=y1_cvt==y_pred
def boolstr_to_floatstr(v):
    if v == 'True':
        return '1'
    elif v == 'False':
        return '0'
    else:
        return v
x = np.vectorize(boolstr_to_floatstr)(x).astype(float)
x
x=np.sum(x,axis=1,keepdims=True)/y1_cvt.shape[1]
x=x.T
np.sum(x,axis=1,keepdims=True)/7