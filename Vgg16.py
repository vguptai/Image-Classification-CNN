import tensorflow as tf
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    
    data_dict = None
    '''
    Load the numpy model file into a dictionary
    '''
    def __init__(self, rgbImagesPlaceholder, fineTuneModel=False, model_path=None):
        self.data_dict = np.load(model_path, encoding='latin1').item()
        self._buildVgg16FeatureExtractor(rgbImagesPlaceholder, fineTuneModel)
        print("The model has been loaded...")

    '''
    The VGG16 network uses mena subtracted images in BGR format as it used OpenCV which loaded image as BGR.
    '''
    def _preprocessImageAsPerVGG16(self,rgbImage):
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgbImage)
        bgrImage = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        return bgrImage

    def getFinalLayer(self):
        return self.pool5
    '''
    Build the model using the loaded weights and keep them fix if "fineTuneModel" is set to false, else
    allow the weights to be updated 
    '''
    def _buildVgg16FeatureExtractor(self, rgbImages, fineTuneModel):
        
        bgrMeanSubtractedImage = self._preprocessImageAsPerVGG16(rgbImages)

        self.conv1_1 = self._conv_layer(bgrMeanSubtractedImage, "conv1_1",fineTuneModel)
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2",fineTuneModel)
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1",fineTuneModel)
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2",fineTuneModel)
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1",fineTuneModel)
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2",fineTuneModel)
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3",fineTuneModel)
        self.pool3 = self._max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1",fineTuneModel)
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2",fineTuneModel)
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3",fineTuneModel)
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1",fineTuneModel)
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2",fineTuneModel)
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3",fineTuneModel)
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')
        
        # self.fc6 = self._fc_layer(self.pool5, "fc6")
        # self.relu6 = tf.nn.relu(self.fc6)

        # self.fc7 = self._fc_layer(self.relu6, "fc7")
        # self.relu7 = tf.nn.relu(self.fc7)

        # self.fc8 = self._fc_layer(self.relu7, "fc8")

        # self.prob = tf.nn.softmax(self.fc8, name="prob")

        # self.data_dict = None

        return self.pool5

        print "VGG16 Feature Extractor Model Built...."


    def _conv_layer(self, input, name,fineTuneModel):
        with tf.variable_scope(name):
            filt = self._get_conv_filter(name,fineTuneModel)
            conv = tf.nn.conv2d(input, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self._get_bias(name,fineTuneModel)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, input, name):
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(input, [-1, dim])
            biases = self._get_bias(name)
            weights = self._get_fc_weight(name)
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def _get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def _get_conv_filter(self, name, fineTuneModel):
    	if fineTuneModel:
            return tf.Variable(self.data_dict[name][0], name="filter")
        else:
            return tf.constant(self.data_dict[name][0], name="filter")

    def _get_bias(self, name, fineTuneModel):
        if fineTuneModel:
            return tf.Variable(self.data_dict[name][1], name="biases")
        else:
    	   return tf.constant(self.data_dict[name][1], name="biases")

    def _max_pool(self, bottom, name):
    	return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def predict(self,rgbImages,session):
        return session.run([self.prob], feed_dict={self.rgbImages: rgbImages})
