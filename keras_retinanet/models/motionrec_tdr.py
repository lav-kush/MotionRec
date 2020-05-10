# -*- coding: utf-8 -*-

"""
Copyright 2020-2021 MotionRec
@Author Lav Kush Kumar

This module implements temporal depth reductionist (TDR) block for background estimation
"""
from keras.layers import Activation, Concatenate,  Subtract, Average, Maximum, Minimum
from keras.layers.convolutional import Conv2D


def background_est_cnn(image_frame):
    """
    implements:
        background estimation using temporal depth reduction

        inputs:
            image_fame: history image frame for background computation
        return: 
            background estimation model

    """
    tdr_layer_1_1 = Conv2D(32, kernel_size = (1, 1), strides = 1, padding = 'same',  name='tdr_layer_1_1', data_format='channels_last')(image_frame)
    tdr_layer_1_1 = Activation('relu')(tdr_layer_1_1)
    tdr_layer_1_2 = Conv2D(32, kernel_size = (3, 3), strides = 1, padding = 'same',  name='tdr_layer_1_2', data_format='channels_last')(image_frame)
    tdr_layer_1_2 = Activation('relu')(tdr_layer_1_2)
    tdr_layer_1_3 = Conv2D(32, kernel_size = (5, 5), strides = 1, padding = 'same',  name='tdr_layer_1_3', data_format='channels_last')(image_frame)
    tdr_layer_1_3 = Activation('relu')(tdr_layer_1_3)
    
    tdr_layer_1 = Average()([tdr_layer_1_1, tdr_layer_1_2, tdr_layer_1_3])
    
    
    tdr_layer_2_1 = Conv2D(16, kernel_size = (1, 1), strides = 1, padding = 'same',  name='tdr_layer_2_1', data_format='channels_last')(tdr_layer_1)
    tdr_layer_2_1 = Activation('relu')(tdr_layer_2_1)
    tdr_layer_2_2 = Conv2D(16, kernel_size = (3, 3), strides = 1, padding = 'same',name='tdr_layer_2_2',  data_format='channels_last')(tdr_layer_1)
    tdr_layer_2_2 = Activation('relu')(tdr_layer_2_2)
    tdr_layer_2_3 = Conv2D(16, kernel_size = (5, 5), strides = 1, padding = 'same', name='tdr_layer_2_3', data_format='channels_last')(tdr_layer_1)
    tdr_layer_2_3 = Activation('relu')(tdr_layer_2_3)
    
    tdr_layer_2 = Average()([tdr_layer_2_1, tdr_layer_2_2, tdr_layer_2_3])
    
    tdr_layer_3_1 = Conv2D(8, kernel_size = (1, 1), strides = 1, padding = 'same', name='tdr_layer_3_1',  data_format='channels_last')(tdr_layer_2)
    tdr_layer_3_1 = Activation('relu')(tdr_layer_3_1)
    tdr_layer_3_2 = Conv2D(8, kernel_size = (3, 3), strides = 1, padding = 'same', name='tdr_layer_3_2',  data_format='channels_last')(tdr_layer_2)
    tdr_layer_3_2 = Activation('relu')(tdr_layer_3_2)
    tdr_layer_3_3 = Conv2D(8, kernel_size = (5, 5), strides = 1, padding = 'same', name='tdr_layer_3_3',  data_format='channels_last')(tdr_layer_2)
    tdr_layer_3_3 = Activation('relu')(tdr_layer_3_3)
    
    tdr_layer_3 = Average()([tdr_layer_3_1, tdr_layer_3_2, tdr_layer_3_3])
    
    tdr_layer_4 = Conv2D(1, kernel_size = (3, 3), padding = 'same', name="TDR_block",  data_format='channels_last')(tdr_layer_3)
    model = Activation('relu')(tdr_layer_4)

    return model
