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
        testX,testY = genericDataSetLoader.getNextTestBatch()
        print('Accuracy:', convNetModel.test(testX,testY))


convNetModel = convNetModel()
saver = tf.train.Saver()
testNeuralNetwork()
