from genericDataSetLoader import *

genericDataSetLoader = genericDataSetLoader(False,"dataset",2,0.8,224,224)
# genericDataSetLoader.loadData()

def nextD():
    x,y = genericDataSetLoader.getNextTrainBatch(5)
    if(x is None):
        print "X is none now"
        return
    print x.shape
    #img1 = Image.fromarray(x[0], 'RGB')
    #img1.show()
    print y

# nextD()
# nextD()
# nextD()
# nextD()
# nextD()

genericDataSetLoader.analyzeDataDistribution()
# oneHotVectors = np.array([[1,0],[1,0],[0,1],[1,0],[0,1]])
# #oneHotVectors = np.array([[1,0]])
# print oneHotVectors
# print oneHotVectors.shape
# labels = np.argmax(oneHotVectors==1,axis=1)
# print labels[:,np.newaxis]
# print labels.shape

# a = [1,3,5,1,3,3,3,5,6,7]
# a = np.array(a)
# print a
# print np.count_nonzero(a==3)