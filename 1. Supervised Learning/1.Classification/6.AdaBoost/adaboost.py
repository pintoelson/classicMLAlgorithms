from numpy import *
import matplotlib.pyplot as plt

def loadSimpData():
    datMat = matrix([
    	[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]
        ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

# dataMatrix, labelMat = loadSimpData()
# dimen = 0
# threshVal = 1.2
# threshIneq = 'lt'

# print(dataMatrix[:,1])
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = 1.0
    return retArray

# print(stumpClassify(dataMatrix,dimen,threshVal,threshIneq))
# retArray = ones((shape(dataMatrix)[0], 1))
# print(retArray)
# if threshIneq == 'lt':
# 	retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
# else
# print(retArray)
D = mat(ones((5,1))/5)
# print(D)
dataArr, classLabels = loadSimpData()
def buildStump(dataArr, classLabels, D):
	dataMatrix = mat(dataArr);	labelMat = mat(classLabels).T
	m,n = shape(dataMatrix) #5x2
	numSteps = 10.0; bestStump = {}; bestClassEst = mat(zeros((m, 1)))
	minError =  inf
	# print(minError)
	for i in range(n):
		rangeMin = dataMatrix[:,i].min();	rangeMax = dataMatrix[:,i].max();	stepSize = (rangeMax-rangeMin)/numSteps
		# print(rangeMax, rangeMin, stepSize)
		for j in range(-1, int(numSteps)+1):
			for inequal in ['lt', 'gt']:
				threshVal = (rangeMin + float(j)*stepSize)
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
				errArr = mat(ones((m,1)))
				errArr[predictedVals == labelMat] = 0
				weightederror = D.T*errArr
				# print("split : %d, thresh : %.2f, thres inequal: %s, weighted error: %.2f" % (i, threshVal, inequal, weightederror))
				if weightederror < minError:
					minError = weightederror
					bestClassEst = predictedVals.copy()
					bestStump['dim'] = i 
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
					# bestStump['we'] = weightederror
	return bestStump, minError, bestClassEst

dataArr, classLabels = loadSimpData()

def adaBoostTrainDS(dataArr, classLabels, numiter = 40):
	weakClassArr = []
	m = shape(dataArr)[0]
	D = mat(ones((m,1))/m)
	# print(D.T)
	aggClassEst = mat(zeros((m, 1)))
	for i in range(40):
	    bestStump,error,classEst = buildStump(dataArr,classLabels,D)
	    alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
	    bestStump['alpha'] = alpha
	    # print(bestStump)
	    weakClassArr.append(bestStump)
	    # print("classEst : ", classEst.T)
	    expon = multiply(-1*alpha*mat(classLabels).T,classEst)
	    # print("expon : ",expon)
	    D = multiply(D,exp(expon))
	    D = D/D.sum()
	    # print("D : ",D.T)
	    aggClassEst += alpha*classEst
	    # print("aggClassEst : ", aggClassEst.T)
	    aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
	    # print("aggErrors: ",aggErrors)
	    errorRate = aggErrors.sum()/m
	    # print("errorRate : ", errorRate)
	    if errorRate == 0.0: break
	return weakClassArr

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print(aggClassEst)
    return sign(aggClassEst)

# print(adaClassify([0,0],adaBoostTrainDS(loadSimpData()[0], loadSimpData()[1], 10)))

def loadDataSet(fileName):     
    numFeat = len(open(fileName).readline().split('\t')) 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat 

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)