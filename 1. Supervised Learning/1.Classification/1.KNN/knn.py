import matplotlib.pyplot as plt
import numpy as np
import operator
import os

def createDataSet():
	group = np.array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def classify(inX, dataset, labels, k):
	dataSetSize = dataset.shape[0]
	diffMat = np.tile(inX, (dataSetSize,1))- dataset
	sqdiffMat = diffMat**2
	sqDistances = sqdiffMat.sum(axis = 1)
	distances = sqDistances**0.5
	sortedDistIndices = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.items() , key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())
	returnMat = np.zeros((numberOfLines,3))
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]
		classLabelVector.append(listFromLine[-1])
		index += 1
	return returnMat, classLabelVector


def autoNorm(dataSet):
	minvals = dataSet.min(0)
	maxvals = dataSet.max(0)
	print(maxvals, minvals)
	ranges = maxvals - minvals
	normDataSet = np.zeros((dataSet.shape[0],dataSet.shape[1]))
	normDataSet = (dataSet - np.tile(minvals, (m,1)))/ np.tile(ranges, (m,1))
	return normDataSet

def img2vector(filename):
	returnVect = np.zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,j+i*32] = int(lineStr[j])
	return returnVect


def handwritingRecognizer():
	hwLabels = []
	trainingFileList = os.listdir('datasets/trainingDigits')
	m =len(trainingFileList)
	trainingMat = np.zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		hwLabels.append(fileStr[0])
		trainingMat[i,:] = img2vector('datasets/trainingDigits/'+fileNameStr)
	testFileList = os.listdir('datasets/testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = fileStr[0]
		vectorUnderTest = img2vector('datasets/testDigits/'+fileNameStr)
		output = classify(vectorUnderTest, trainingMat, hwLabels, 3)
		print("Actual : "+classNumStr+" Predicted : "+output)
		if(classNumStr != output):	errorCount += 1
	print("Success rate: "+ str((mTest-errorCount)*100/float(mTest)))		#Success rate: 98.94291754756871

# handwritingRecognizer()
# 98.94291754756871
def useSVM():
	hwLabels = []
	trainingFileList = os.listdir('datasets/trainingDigits')
	m =len(trainingFileList)
	trainingMat = np.zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		hwLabels.append(fileStr[0])
		trainingMat[i,:] = img2vector('datasets/trainingDigits/'+fileNameStr)
	from sklearn import svm
	clf = svm.SVC()
	clf.fit(trainingMat, hwLabels)
	# errorCount = 0
	# vectorUnderTest = img2vector('datasets/testDigits/1_0.txt')
	# print(clf.predict(vectorUnderTest))
	testFileList = os.listdir('datasets/testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = fileStr[0]
		vectorUnderTest = img2vector('datasets/testDigits/'+fileNameStr)
		predClass = clf.predict(vectorUnderTest)[0]
		print("Actual : "+ fileNameStr[0] +" Predicted : "+ predClass)
		if(int(fileNameStr[0]) != int(predClass)):	errorCount += 1
	print("Success rate: "+ str(((mTest-errorCount)*100)/mTest) )			#Success rate: 98.5200845665962
