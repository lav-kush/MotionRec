"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

implements :
    MotionRec generator script
    Return list of input list to 
"""

import numpy as np
import random
import warnings
import threading
import keras
import cv2

from ..utils.anchors import (
    anchor_targets_bbox,
    anchors_for_shape,
    guess_shapes
)
from ..utils.config import parse_anchor_parameters
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from ..utils.transform import transform_aabb


class Generator(keras.utils.Sequence):
    """ Abstract generator class.
    """

    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        depth = 10,
        group_method='none',  # one of 'none', 'random', 'ratio'
        shuffle_groups=False,
        image_min_side=608,
        image_max_side=608,
        transform_parameters=None,
        compute_anchor_targets=anchor_targets_bbox,
        compute_shapes=guess_shapes,
        preprocess_image=preprocess_image,
        config=None
    ):
        """ Initialize Generator object.

        Args
            transform_generator    : A generator used to randomly transform images and annotations.
            batch_size             : The size of the batches to generate.
            depth                  : Image frame depth.
            group_method           : Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
            compute_shapes         : Function handler for computing the shapes of the pyramid for a given input.
            preprocess_image       : Function handler for preprocessing an image (scaling / normalizing) for passing through a network.
        """
        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.depth                  = int(depth)
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes         = compute_shapes
        self.preprocess_image       = preprocess_image
        self.config                 = config    

        self.group_index = 0
        self.lock        = threading.Lock()
        self.group_images()

        # validate depth and image frames count
        if len(self.image_names) < self.depth:
                self.depth = len(self.image_names)

        # index of image frame in video
        if self.depth: self.group_index = self.depth
        else: self.group_index = int(depth)

        # history image frames
        self.image_frame = self.compute_image_frame_with_depth()

        # current file in train/test folder
        self.current_file = '/'.join(self.image_names[self.depth].split('/')[:-1])

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """ Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def convert_RGB_to_GRAY(self,image):
        """
        RGB to GRAY conversion
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def compute_image_frame_with_depth(self):
        """
        Create history image frames
        """
        return [ self.convert_RGB_to_GRAY( self.resize_image(self.load_image(image_index))[0] ) for image_index in range(self.depth)]

    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert(isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """ Randomly transforms each image and its annotations.
        """

        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image)
        image, image_scale= self.resize_image(image)
        annotations['bboxes'] *= image_scale

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = sorted(list(range(self.size())))
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(2))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def generate_anchors(self, image_shape):
        anchor_params = None
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = parse_anchor_parameters(self.config)
        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(2))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        return list(batches)

    def load_image_frame(self, index):
        """
            load image frame
        """
        if index + self.depth >= len(self.image_names):
            index = 0
            self.group_index = self.depth + 1
        self.image_frame = [ self.resize_image(self.convert_RGB_to_GRAY(self.load_image(image_index)))[0] for image_index in range(index, index + self.depth)]

    def compute_input_output(self, index):
        """ Compute inputs and target outputs for the network.
        """
        new_image_color = self.load_image(index[0])
        new_image_gray = [self.convert_RGB_to_GRAY(new_image_color)]
        new_image_color = self.resize_image(new_image_color)[0]
        annotations_group = self.load_annotations_group(index)

        # check validity of annotations
        new_image_gray, annotations_group = self.filter_annotations(new_image_gray, annotations_group, index)

        # perform preprocessing steps
        new_image_gray, annotations_group = self.preprocess_group(new_image_gray, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(new_image_gray)

        # compute network targets
        targets = self.compute_targets(new_image_gray, annotations_group)

        return self.return_output_of_generator(new_image_color, inputs[0]), targets

    def return_output_of_generator(self, new_image_color, new_image_gray):
        """ Return generator input list
        """
        self.image_frame = self.image_frame[1:]
        self.image_frame.append(new_image_gray)
        inputs = np.array(self.image_frame)

        new_image_gray = np.expand_dims(new_image_gray,axis=0)
        new_image_color = np.expand_dims(new_image_color,axis=0)

        # element median of history images
        median_image = np.expand_dims(np.median(inputs, axis=0) ,axis=0)

        new_image_gray = np.expand_dims(np.rollaxis(new_image_gray,0,3), axis=0)
        median_image = np.expand_dims(np.rollaxis(median_image,0,3), axis=0)
        inputs = np.expand_dims(np.rollaxis(inputs,0,3), axis=0)
        return [new_image_color, new_image_gray, median_image, inputs]
      
    def __len__(self):
        """
        Number of batches for generator.
        """
        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating list of generator inputs list.
        """
        # change folder when index match new csv file data index
        if self.group_index in self.folder_new_image_index:
            # randomly select new csv file name (folder containing image files)
            new_folder_selected_index = self.folder_new_image_index[random.randint(0,len(self.folder_new_image_index)-1)]
            self.group_index = new_folder_selected_index
            self.load_image_frame(self.group_index)
            self.group_index += int( self.depth)
            print ('\n','/'.join(self.image_names[new_folder_selected_index].split('/')[:-1]) , 'folder loaded')

        with self.lock:
            self.group_index = (self.group_index + 1) % len(self.groups)
            image_index = self.groups[self.group_index]

        return self.compute_input_output(image_index)