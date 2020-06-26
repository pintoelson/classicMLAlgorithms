import math
import numpy as np
import random

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

# dataMatIn = loadDataSet()[0]
# classLabels = loadDataSet()[1]

def gradAscent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn)             #convert to NumPy matrix
	labelMat = np.mat(classLabels).transpose() #convert to NumPy matrix
	m,n = np.shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = np.ones((n,1))
	for k in range(maxCycles):              #heavy on matrix operations
	    h = sigmoid(dataMatrix*weights)     #matrix mult
	    # print(dataMatrix*weights)
	    # print(h)
	    error = (labelMat - h)              #vector subtraction
	    # print(error)
	    weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
	    # print(weights)
	    # break
	return weights

# dataMatrix = np.array(loadDataSet()[0])
# classLabels = loadDataSet()[1] 
def stocGradAscent(dataMatrix, classLabels):
	dataMatrix = np.array(dataMatrix)
	m,n = np.shape(dataMatrix)
	alpha = 0.01
	weights = np.ones(n)
	# print(weights)
	for i in range(m):
		# print(dataMatrix[i])
		h = sigmoid(sum(dataMatrix[i]*weights))
		# print(h)
		error = classLabels[i] - h
		weights = weights + error * alpha * dataMatrix[i]
	return weights

# dataMatrix = np.array(loadDataSet()[0])
# classLabels = loadDataSet()[1]
numIter = 150
def stocGradAscentImproved(dataMatrix, classLabels , numIter = 150):
	dataMatrix = np.array(dataMatrix)
	m,n = np.shape(dataMatrix)
	weights = np.ones(n)
	# print(weights)
	for j in range(numIter):
		dataIndex = list(range(1,m+1))
		# print(dataIndex)
		for i in range(m):
			alpha = 15/((i+j)+0.1)
			randIndex = int(random.uniform(0,len(dataIndex)))
			# print(dataIndex)
			# print(randIndex)
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			# print(h)
			error = classLabels[randIndex] - h
			# print(error)
			weights = weights + error*alpha*dataMatrix[randIndex]
			del(dataIndex[randIndex])
	# 	break
	# break	
	return weights

def plotBestFit(wei):
	import matplotlib.pyplot as plt
	weights = wei
	dataMat , labelMat = loadDataSet()
	dataArr = np.array(dataMat)
	# print(dataArr)
	n = np.shape(dataArr)[0]
	# print(n)
	xcord1 = []	;	ycord1 = []
	xcord2 = []	;	ycord2 = []
	for i in range(n):
		if labelMat[i] == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1 , s=15 , c = 'red')
	ax.scatter(xcord2, ycord2 , s=30 , c ='green')
	x = np.arange(-3.0, 3.0 , 0.001)
	y = (-weights[0] - weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.show()


# plotBestFit(stocGradAscentImproved(loadDataSet()[0], loadDataSet()[1]))

def classifyVector(inX , weights):
	prob = sigmoid(sum(inX*weights))
	if(prob >0.5):	return 1.0
	else:	return 0.0	

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = [] ; trainingLabels = []
	for  line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		# print(lineArr)
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))
		# break
	trainWeights = stocGradAscentImproved(trainingSet, trainingLabels, 1000)
	errorCount = 0	; numTestVec = 0.0	; noOfSamples = 0
	for line in frTest.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		# print("Actual : "+ str(float(currLine[-1])) +" Predicted : "+str(classifyVector(np.array(lineArr) ,trainWeights)))
		if int(currLine[-1]) != int(classifyVector(np.array(lineArr) ,trainWeights)):
			errorCount = errorCount + 1
		noOfSamples = noOfSamples + 1
	# print(noOfSamples)
	# print(errorCount)
	successRate = float((noOfSamples - errorCount)*100/noOfSamples)
	# print("Successful Predictions : "+ str(successRate))
	return successRate

def multiTest(numTests):
	succSum = 0.0
	for k in range(numTests):
		tempSucc = colicTest()
		succSum += tempSucc
		print("Test #"+str(k+1)+" : Success Rate is "+str(tempSucc))
	print("\n Average Success Rate over " +str(numTests)+ " tests : "+str(succSum/numTests))
# colicTest()

multiTest(5)