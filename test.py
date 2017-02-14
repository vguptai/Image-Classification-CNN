from genericDataSetLoader import *

genericDataSetLoader = genericDataSetLoader("dataset",2,0.8,224,224)
genericDataSetLoader.loadData()

def nextD():
    x,y = genericDataSetLoader.getNextTrainBatch(5)
    if(x is None):
        print "X is none now"
        return
    print x.shape
    #img1 = Image.fromarray(x[0], 'RGB')
    #img1.show()
    print y

nextD()
nextD()
nextD()
nextD()
nextD()
