# -*- coding: utf-8 -*-

"""
Copyright 2020-2021 MotionRec
@Author Lav Kush Kumar

This module implements 2D residual models, and background estimation
"""

import keras.backend
import keras.layers
import keras.models
import keras.regularizers

import keras_resnet.blocks
import keras_resnet.layers

from .residual_block import bottleneck_2d
from .motionrec_tdr import background_est_cnn

from keras.layers import Activation, Concatenate,  Subtract, Average, Maximum, Minimum
from keras.layers.convolutional import Conv2D


def rec2DNetwork(inputs, blocks, block, include_top=True, current_image_pyramid = False,  classes=1000, freeze_bn=True, numerical_names=None, *args, **kwargs):
    """
    Constructs a `keras.models.Model` object using the given block count.

    :param 
        inputs: input tensor (e.g. an instance of `keras.layers.Input`)
        blocks: the network’s residual architecture
        block: a residual block (e.g. an instance of `residual_block.basic_2d`)
        include_top: if true, includes classification layers
        classes: number of classes to classify (include_top must be true)
        freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
        numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters

    :return 
        model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    """
    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    if not current_image_pyramid:
        new_image_color, new_image_gray, median_image, image_frame = inputs

        background_est = background_est_cnn(image_frame)
        x = Concatenate(axis = -1)([background_est, new_image_color, median_image, new_image_gray])
        layer_name =''
    else: 
        x = inputs
        layer_name = '_new_image'


    x = keras.layers.ZeroPadding2D(padding=3, name="padding_conv1"+layer_name)(x)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1"+layer_name)(x)
    x = keras.layers.Activation("relu", name="conv1_relu"+layer_name)(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1"+layer_name)(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, layer_name = layer_name, numerical_name=(block_id > 0 and numerical_names[stage_id]), freeze_bn=freeze_bn)(x)

        features *= 2

        outputs.append(x)

    if include_top:
        assert classes > 0

        x = keras.layers.GlobalAveragePooling2D(name="pool5"+layer_name)(x)
        x = keras.layers.Dense(classes, activation="softmax", name="fc1000"+layer_name)(x)

        return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
    else:
        # Else output each stages features
        return keras.models.Model(inputs=inputs, outputs=outputs, *args, **kwargs)


def rec2DNet(inputs, blocks=None, include_top=True,current_image_pyramid=False, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet50 specifications.

    :param 
        inputs: input tensor (e.g. an instance of `keras.layers.Input`)
        blocks: the network’s residual architecture
        include_top: if true, includes classification layers
        classes: number of classes to classify (include_top must be true)

    :return 
    model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    """
    if blocks is None:
        blocks = [3, 4, 6, 3]
    numerical_names = [False, False, False, False]

    return rec2DNetwork(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top, current_image_pyramid=current_image_pyramid, classes=classes, *args, **kwargs)