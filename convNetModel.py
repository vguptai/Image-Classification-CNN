import tensorflow as tf
from config import *
from Vgg16 import *

class convNetModel:

    optimizer = None
    accuracy = None
    cost = None
    prediction = None
    correct = None
    x = None
    y = None
    keep_rate = None

    def __init__(self,usePretrainedNetwork=False,fineTunePretrainedModel=False):
        self.x = tf.placeholder('float', [None, imageSizeX,imageSizeY,numChannels])
        self.y = tf.placeholder('float')
        self.keep_rate = tf.placeholder(tf.float32)
        self._setupNetwork(usePretrainedNetwork,fineTunePretrainedModel)

    def _conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _maxpool2d(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def _buildNetwork(self):
	weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, numChannels, 32])),
                   'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                   'W_fc': tf.Variable(tf.random_normal([imageSizeX/4 * imageSizeY/4 * 64, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, n_classes]))}

	biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
                  'b_conv2': tf.Variable(tf.random_normal([64])),
                  'b_fc': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([n_classes]))}


        conv1 = tf.nn.relu(self._conv2d(self.x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = self._maxpool2d(conv1)

        conv2 = tf.nn.relu(self._conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = self._maxpool2d(conv2)

        conv3 = tf.nn.relu(self._conv2d(conv2,weights['W_conv3'])+ biases['b_conv3'])
        conv3 = self._maxpool2d(conv3)

        fc = tf.reshape(conv3, [-1, imageSizeX/8 * imageSizeY/8 * 64])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, weights['out']) + biases['out']

        return output

    def _buildNetworkFromPretrainedVGG16(self,fineTunePretrainedModel):

            weights = {'out': tf.Variable(tf.random_normal([1024, n_classes]))}
            biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

            vgg16FeatureExtrator = Vgg16(self.x,fineTunePretrainedModel, pretrainedModelPath)
            pretrainedNetwork = vgg16FeatureExtrator.getFinalLayer()

            # Here, we are converting the output from the pretrained network [num_images,reduced_xdim,reduced_ydim,final_num_filters]
            # into [num_images,reduced_xdim * reduced_ydim * final_num_filters]
            shape = pretrainedNetwork.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            fc  = tf.reshape(pretrainedNetwork, [-1, dim])

            # First fully connected layer
            w_fc_1 = tf.Variable(tf.random_normal([dim, 1024]))
            b_fc_1 = tf.Variable(tf.random_normal([1024]))
            fc = tf.nn.relu(tf.matmul(fc, w_fc_1) + b_fc_1)
            fc = tf.nn.dropout(fc, self.keep_rate)

            # Output layer
            output = tf.matmul(fc, weights['out']) + biases['out']
            return output

    def _setupNetwork(self,usePretrainedNetwork,fineTunePretrainedModel):

        if usePretrainedNetwork:
            self.prediction = self._buildNetworkFromPretrainedVGG16(fineTunePretrainedModel)
        else:
            self.prediction = self._buildNetwork()

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        self.correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))


    def train(self,sess,trainX,trainY):
        return sess.run([self.optimizer, self.cost], feed_dict={self.x: trainX, self.y: trainY, self.keep_rate: training_keep_rate})

    def test(self,testX,testY):
        return self.accuracy.eval({self.x: testX, self.y: testY, self.keep_rate: testing_keep_rate})
