from numpy import *

def loadDataSet(fileName):      
    numFeat = len(open(fileName).readline().split('\t')) - 1 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# print(loadDataSet('ex0.txt')[0])
# xArr = loadDataSet('ex0.txt')[0]
# yArr = loadDataSet('ex1.txt')[1]

def standRegres(xArr, yArr):
    xMat = mat(xArr)
    # print(shape(xMat))
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    # print(xTx)
    # print(linalg.det(xTx))
    if linalg.det(xTx) == 0.0:
        print("Linear Regression not possible for this dataset.")
        return
    ws = xTx.I * (xMat.T*yMat)
    # print(xTx.I)
    # print(xMat.T*yMat)
    return ws

# xArr = loadDataSet('ex0.txt')[0]
# yArr = loadDataSet('ex1.txt')[1]

# weights = standRegres(xArr, yArr)
# print(weights)
# print(1.000000*weights[0]+0.925577*weights[1])

def drawLinearRegressLine(xArr, yArr):
    xMat, yMat = mat(xArr), mat(yArr)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax =fig.add_subplot()
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T.flatten().A[0])
    weights = standRegres(loadDataSet('ex0.txt')[0], loadDataSet('ex0.txt')[1])
    yPred = xMat * weights
    ax.plot(xMat[:,1], yPred)
    plt.show()


# xArr = loadDataSet('ex0.txt')[0]
# yArr = loadDataSet('ex0.txt')[1]
# drawLinearRegressLine(loadDataSet('ex0.txt')[0], loadDataSet('ex0.txt')[1])

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("Linear Regression not possible for given parameters. Try to change value of k")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def drawRegresLineLWLR(xArr, yArr):
    yHat = lwlrTest(xArr,xArr,yArr,0.01)
    xMat, yMat = mat(xArr), mat(yArr)
    srtInd = xMat[:,1].argsort(0)
    # print(xMat)
    # print(srtInd)
    xSort = xMat[srtInd][:,0,:]
    # print(xSort)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax =fig.add_subplot()
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T.flatten().A[0])
    ax.plot(xSort[:,1], yHat[srtInd], color = 'red')
    # ax.plot([1,2,3], [4,1,6])
    plt.show()

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

# xArr = loadDataSet('ex0.txt')[0]
# yArr = loadDataSet('ex0.txt')[1]
# drawRegresLineLWLR(xArr, yArr)


def predictAbaloneAge(k = 0.1):         #k in kernel size
    abX, abY = loadDataSet('abalone.txt')
    yHat = lwlrTest(abX, abX, abY, k)         #Can change range of data-set provided
    # print(rssError(abY[0:99], yHat.T))
    print("Error estimate: "+str(rssError(abY, yHat.T)))
    return abX, yHat

# predictAbaloneAge(1)

def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

# xArr, yArr = loadDataSet('abalone.txt')
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    # print(yMat)
    # print(xMat)
    yMean = mean(yMat,0)
    # print(yMean)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    # #regularize X's
    # print(yMat)
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    # print(xMeans)
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    # print(xVar)
    print("xvar :"+str( xVar))
    xMat = (xMat - xMeans)/xVar
    # print(xMat)
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    # print(shape(wMat))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def plotRidgeWeights():
    xArr, yArr = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(xArr, yArr)
    # print(ridgeWeights)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

def predictYRidge(xArr, yArr, wantToPlot, logLambda):
    xMat, yMat = mat(xArr), mat(yArr).T
    ws = ridgeRegres(xMat,yMat,exp(logLambda))
    # print(shape(ws))
    yPred = xMat * ws 
    # print(yPred)
    if(wantToPlot == 1):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax =fig.add_subplot()
        ax.scatter(xMat[:,1].flatten().A[0], yMat.T.flatten().A[0])
        ax.plot(xMat[:,1], yPred, color = 'red')
        plt.show()
    return yPred

# xArr, yArr = loadDataSet('ex0.txt')
# predictYRidge(xArr, yArr, 1, 1)

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    n=shape(xMat)[1]
    #returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
    return ws.T

