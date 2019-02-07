import os
import sys
import math
import pandas as pd
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

#this is a simple [(1,1),(2,1),(3,1)..(10,1)] data
#y[3,5,7,9..]
#100% fit

trainSet_X=np.array(range(1,11))
trainSet_y=trainSet_X**2+trainSet_X*2+1
trainSet_X=np.array([(i,1) for i in range(1,11)])
trainSet_X=trainSet_X.reshape(-1,2)
trainSet_y=trainSet_y.reshape(-1,1)

'''
#this is (100*6) random data
#y 100*1 random
#not 100% fit when f(i1)===f(i2) due to random initiation
np.random.seed(2)
trainSet_X=np.random.rand(100,6)
trainSet_y=np.random.rand(100,1)
trainSet_y=trainSet_y.reshape(-1,1)

#this portfolio data
#not 100% thought
#2 instances when f(i1)===f(i2) 

trainSet=pd.read_csv("trainSet.csv")
trainSet_y=trainSet["runtime_ham10"]
trainSet_y=np.array(trainSet_y)
trainSet_y.reshape(-1,1)
trainSet_X=trainSet.iloc[:,:-8]
'''


number_of_bin=10
max_neigh = range(1, 5, 1)
knn_scores = []
print(str(number_of_bin) +" bins validation for each depth")
for k in max_neigh:
	#show n fold cross_val_score at current depth

	kNeigh =KNeighborsRegressor(n_neighbors=k)
	loss = -cross_val_score(kNeigh,trainSet_X, trainSet_y, cv=number_of_bin, scoring="neg_mean_squared_error")
	print(k,":",loss)
	#kNN_scores.append(loss.mean())

	#only first bins val_score
	kNeigh =KNeighborsRegressor(n_neighbors=k)
	numof1st=math.ceil(len(trainSet_X)/float(number_of_bin))*(number_of_bin-1) 
	kNeigh= kNeigh.fit(trainSet_X[:numof1st], trainSet_y[:numof1st])
	y_=kNeigh.predict(trainSet_X[numof1st:])
	knn_scores.append(mean_squared_error(trainSet_y[numof1st:], y_))
print("1st bin validation for each depth")
#to find how they use first bins
#my way: ceil(all/float(bins))*(bins-1), something same
print(knn_scores)

numof1st=math.ceil(len(trainSet_X)/float(number_of_bin))*(number_of_bin-1)
max_neigh = range(1, min(200,numof1st))

dt_scores = [] #avg cross validation
dt_scores_1stbin=[] # 1st bin validation
dt_scores_full=[] #self train error
for k in max_neigh:
	#print(k)
	regr_k =KNeighborsRegressor(n_neighbors=k)
	loss = -cross_val_score(regr_k, trainSet_X, trainSet_y, cv=number_of_bin, scoring="neg_mean_squared_error")
	#print(loss)
	dt_scores.append(loss.mean())


	numof1st=math.ceil(len(trainSet_X)/float(number_of_bin))*(number_of_bin-1) 
	dtModel=KNeighborsRegressor(n_neighbors=k)
	dtModel= dtModel.fit(trainSet_X[:numof1st], trainSet_y[:numof1st])
	y_=dtModel.predict(trainSet_X[:numof1st])
	dt_scores_1stbin.append(mean_squared_error(trainSet_y[:numof1st], y_))


	dtModel=KNeighborsRegressor(n_neighbors=k)
	dtModel= dtModel.fit(trainSet_X, trainSet_y)
	
	y_=dtModel.predict(trainSet_X)
	resut_cmp=zip(y_,trainSet_y)
	for each in resut_cmp:
		#print "not fit" in self training
		if each[0]!=each[1]:
			pass
			#print(each)
	dt_scores_full.append(mean_squared_error(trainSet_y, y_))
print(str(number_of_bin) +"bins validation average for each depth")
print(dt_scores)
print("self training error for each depth")
print(dt_scores_full)
print("1st bin validation")
print(dt_scores_1stbin)


print("bestDepthDT:")
bestscoreDT,bestDepthDT=sorted(list(zip(dt_scores,max_neigh)))[0]
print(bestDepthDT)
bestscoreDT,bestDepthDT=sorted(list(zip(dt_scores_full,max_neigh)))[0]
print(bestDepthDT)
bestscoreDT,bestDepthDT=sorted(list(zip(dt_scores_1stbin,max_neigh)))[0]
print(bestDepthDT)




numof1st=math.ceil(len(trainSet_X)/float(number_of_bin))*(number_of_bin-1) 

dtModel=KNeighborsRegressor(n_neighbors=bestDepthDT)
dtModel= dtModel.fit(trainSet_X[:numof1st], trainSet_y[:numof1st])
y_=dtModel.predict(trainSet_X[numof1st:])

resut_cmp=zip(y_,trainSet_y[numof1st:])
for each in resut_cmp:
	pass
	#print(each)








#print "DTscoring:",dt_scores
plt.subplot(2,1,1)
plt.plot(max_neigh, dt_scores,label="KNN")
plt.xlabel('Value of k, avg validation')
plt.ylabel('Cross-Validated MSE')
'''
plt.subplot(3,1,2)
plt.plot(max_neigh, dt_scores_full,label="KNN")
plt.xlabel('Value of k')
plt.ylabel('Cross-Validated MSE')
'''
plt.subplot(2,1,2)
plt.plot(max_neigh, dt_scores_1stbin,label="KNN")
plt.xlabel('Value of k, avg training')
plt.ylabel('Cross-Validated MSE')

plt.show()

