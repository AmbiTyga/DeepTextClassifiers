# -*- coding: utf-8 -*-

"""

@author: Ambesh Shekhar

@contact: ambesh.sinha@gmail.com

@file: kmaxpooling.py

@time: 2019/2/8 13:08

@desc:

"""
from tensorflow.keras.layers import Flatten, Layer, InputSpec
import tensorflow as tf

class KMaxPooling(Layer):
    """
    Implemetation of temporal k-max pooling layer, which was first proposed in Kalchbrenner et al.
    [http://www.aclweb.org/anthology/P14-1062]
    "A Convolutional Neural Network for Modelling Sentences"
    This layer allows to detect the k most important features in a sentence, independent of their
    specific position, preserving their relative order.
    """
    def __init__(self, k=1, sorted=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
        self.sorted = sorted

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k, input_shape[2])

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_inputs = tf.transpose(inputs, [0, 2, 1])
        
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_inputs, k=self.k, sorted=self.sorted)[0]
        
        # return flattened output
        return tf.transpose(top_k, [0,2,1])
