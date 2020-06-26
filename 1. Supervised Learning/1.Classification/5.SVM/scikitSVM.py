from sklearn import svm

# Done using library scikit-learn

def loadDataSet(filename):
	dataMat = []
	labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[-1]))
	return dataMat, labelMat

def plotPoints(dataMat):
	# print(dataMat)
	x = []
	y = []
	for i in range(len(dataMat)):
		x.append(dataMat[i][0])
		y.append(dataMat[i][1])
	print(x)
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x, y , s=15 , c = 'red')
	plt.show()

# print(loadDataSet('testSet.txt')[1])
def testClassifier(x, y):
	clf = svm.SVC()
	clf.fit(x, y)
	errorCount = 0
	for i in range(len(x)):
		predClass =clf.predict([x[i]])
		print("Actual : "+ str(float(predClass[0])) +" Predicted : "+str(float(y[i])))
		if (float(predClass[0]) != float(y[i])):
			errorCount += 1 
	print("Success rate: "+ str(((len(x)-errorCount)*100)/len(x)) )
	print("Number of errors: "+str(errorCount))
# x = loadDataSet('testSet.txt')[0]
# y = loadDataSet('testSet.txt')[1]
# testClassifier(x, y)

plotPoints(loadDataSet('testSet.txt')[0])