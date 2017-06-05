import tensorflow as tf
from genericDataSetLoader import *
from config import *
from convNetModel import *
genericDataSetLoader = genericDataSetLoader(False,"dataset",n_classes,testTrainSplit,imageSizeX,imageSizeY)
genericDataSetLoader.loadData()

def testNeuralNetwork():
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path+" :Testing this checkpoint...")
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        calculateTestAccuracy()

def calculateTestAccuracy():
    genericDataSetLoader.resetTestBatch()
    batchAccuracies = []
    while(True):
        testX, testY = genericDataSetLoader.getNextTestBatch(batch_size)
        if(testX is None):
            break
        acc = convNetModel.test(testX,testY)
        batchAccuracies.append(acc)
        print "Accuracy of test batch..."+str(acc)
    #testX = np.reshape(testX, (-1, imageSizeX, imageSizeY, numChannels))
    print('Accuracy:', sum(batchAccuracies) / float(len(batchAccuracies)))

convNetModel = convNetModel(usePretrainedNetwork,fineTunePretrainedModel)
saver = tf.train.Saver()
testNeuralNetwork()
