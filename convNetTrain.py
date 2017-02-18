import tensorflow as tf
from genericDataSetLoader import *
from config import *
import os
from convNetModel import *

genericDataSetLoader = genericDataSetLoader(False,"dataset",n_classes,testTrainSplit,imageSizeX,imageSizeY)
genericDataSetLoader.loadData()

def calculateTestAccuracy():
    testX,testY = genericDataSetLoader.getNextTestBatch()
    #testX = np.reshape(testX, (-1, imageSizeX, imageSizeY, numChannels))
    print('Accuracy:', convNetModel.test(testX,testY))


def restoreFromCheckPoint(sess,saver):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path+" :Restoring from a checkpoint...")
        saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        start = global_step.eval() # get last global_step
        start = start+1
    else:
        print "Starting fresh training..."
        start = global_step.eval() # get last global_step
    return start

def trainNeuralNetwork():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #Restore the model from a previous checkpoint if any and get the epoch from which to continue training
        start = restoreFromCheckPoint(sess,saver)
        print "Start from:"+str(start)+"/"+str(numEpochs)

        #Training epochs
        for epoch in range(start,numEpochs):
            epoch_loss = 0

            genericDataSetLoader.resetBatch()

            while(True):
                epoch_x, epoch_y = genericDataSetLoader.getNextTrainBatch(batch_size)
                if(epoch_x is None):
                    break
                _, c = convNetModel.train(sess,epoch_x,epoch_y)
                epoch_loss += c

            global_step.assign(epoch).eval()
            saver.save(sess,'model/data-all.chkp',global_step=global_step)
            print('Epoch', epoch, 'completed out of', numEpochs, 'loss:', epoch_loss)

            #Get the validation/test accuracy
            calculateTestAccuracy()

#if not os.path.exists(ckpt_dir):
#    os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

convNetModel = convNetModel()
saver = tf.train.Saver()
trainNeuralNetwork()
