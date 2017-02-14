import tensorflow as tf
from genericDataSetLoader import *
from config import *

genericDataSetLoader = genericDataSetLoader(True,"dataset",n_classes,testTrainSplit,imageSizeX,imageSizeY)
genericDataSetLoader.loadData()

x = tf.placeholder('float', [None, imageSizeX,imageSizeY,numChannels])
y = tf.placeholder('float')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, numChannels, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([imageSizeX/4 * imageSizeY/4 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    #x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(numEpochs):
            epoch_loss = 0
            #reset the offset of the batch
            genericDataSetLoader.resetBatch()
            while(True):
                epoch_x, epoch_y = genericDataSetLoader.getNextTrainBatch(batch_size)
                if(epoch_x is None):
                    break
                #img1 = Image.fromarray(epoch_x[1])
                #img1.show()
                epoch_x = np.reshape(epoch_x, (-1, imageSizeX, imageSizeY, numChannels))
                # print epoch_y[1]
                # return
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', numEpochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        testX,testY = genericDataSetLoader.getNextTestBatch()
        testX = np.reshape(testX, (-1, imageSizeX, imageSizeY, numChannels))
        print('Accuracy:', accuracy.eval({x: testX, y: testY}))


train_neural_network(x)
