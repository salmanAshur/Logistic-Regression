import numpy as np


def sigmoid(X):
	return (1.0 / (1 + np.exp(-X)))
	
def prediction(X,Y,W,b):

	print("shape: ",X.shape,Y.shape,W.shape)
	for i in range(500):
		z = np.dot(W,X[i].T)+b
		activation = sigmoid(z)
		label = 0 if activation <0.5 else 1
		print("activation={}; predicted_label={}, true_label={}".format(activation, label, Y[i]))
def logistic_regressor():
	b = 1
	split = 800
	alpha = 0.1
	Adata_set = np.genfromtxt("Data-set_2.csv",delimiter = ',')
	np.random.shuffle(Adata_set)
	

	#Separating Training Data ans Test data
	train,test = Adata_set[:split,:],Adata_set[split:,:]
	
	W = np.array([0.1,0.2,0.4,0.6],dtype = 'f')
	W = np.array([W])
	
	#Now slicing the input data and output data
	X,Y = train[:,:-1],train[:,-1]
	print("Shape of the X AND Y(W) before transpose ",X.shape,"  ",Y.shape)
	X = X.T
	Y = np.array([Y])
	m = np.prod(X.shape)
	print("Shape of the X AND Y(W) after transpose ",X.shape,"  ",Y.shape)

	for i in range(10000):
		Z = np.dot(W,X) + b

		#calculating the sigmoid
		A = sigmoid(Z)
		
		#calculating the derivative W.R to W
		dz = A - Y
		error = np.sum(dz**2)
		#print("Training error is ",error)
		#calculating the derivative 
		dw = np.dot(X,Z.T)/m

		#calculating derivative W.R to b
		db = np.sum(dz)/m
		
		
		#updating weights
		W -= alpha*dw.T
		#print("Weights after ",i," iteration  is ",W)
		#updating broadcas parameter
		b -= alpha*(db)
		#print("b after ",i," iteration  is ",b)
	New_X, New_Y = test[:,:-1],test[:,-1]
	New_Y = np.array([New_Y])
	prediction(New_X,New_Y.T,W,b)
	

logistic_regressor()