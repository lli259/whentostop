import numpy as np
import math
def cross_val_score2(trainSet_X, trainSet_y, bins, scoring="neg_mean_squared_error"):
	ret=[]
	blinlistX=[]
	blinlisty=[]
	numinbins=math.ceil(len(trainSet_X)/float(bins))
	for i in range(bins-1):
		blinlistX.append(trainSet_X[i*numinbins:(i+1)*numinbins])
		blinlisty.append(trainSet_y[i*numinbins:(i+1)*numinbins])
	blinlistX.append(trainSet_X[-numinbins:])
	blinlisty.append(trainSet_y[-numinbins:])
	for i in range(bins):
		vindex=(i-1)%10
		vxdata=blinlistX[vindex]
		vydata=blinlisty[vindex]
		txdata=np.array([])
		tydata=np.array([])
		for j in range(9):
			txdata=np.concatenate((txdata,blinlistX[(j+i)%10]),axis=0) 
			tydata=np.concatenate((tydata,blinlisty[(j+i)%10]),axis=0) 
		print(vxdata,vydata,txdata,tydata)
		
		'''
		dtModel= dtModel.fit(txdata,tydata)
		y_=dtModel.predict(vxdata)
		dt_scores.append(mean_squared_error(vydata, y_))
		ret.append()
		'''
	return ret

cross_val_score2(np.array(range(20)),np.array(range(20)),10)
