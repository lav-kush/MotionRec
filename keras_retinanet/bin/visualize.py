#!/usr/bin/env python

"""
MotionRec Layer visualization
"""

import argparse
import cv2
import numpy as np
import os
import seaborn as sns
import sys

from matplotlib import pyplot as plt
from scipy import ndimage
from keras.preprocessing import image as KImage

import keras
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
from keras.utils import plot_model

# For using single gpu, enter gpu id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sns.set()

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval import evaluate
from ..utils.keras_version import check_keras_version


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for evaluation.
    """
    if args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.classes,
            csv_path = args.csv_path,
            depth = args.depth,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    parser.add_argument('model',              help='Path to RetinaNet model.')

    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=608)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=608)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--depth',      help='Image frame depth.', default=10, type=int)
    parser.add_argument('--csv_path',    help='Path to store csv files for layer visualization ', default='./visualize/backdoor/csv/')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_false', default='False')

    parser.add_argument('--save-path',        help='Path for saving images with visualization', default='./visualize/backdoor/')
    parser.add_argument('--layer',              help='Name of the CNN layer to visualize.', default='TDR_block')
    parser.add_argument('--layer_size',      help='CNN layer size.', default=608, type=int)

    return parser.parse_args(args)

def main(args=None):
    """ Entry Point
    """
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.save_path+'input/'):
        os.makedirs(args.save_path+'input/')
    if not os.path.exists(args.save_path+args.layer+'/output/'):
        os.makedirs(args.save_path+args.layer+'/output/')

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    print('Creating generator, this may take a second...')
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model, this may take a second...')
    model = keras_retinanet.models.backbone('resnet50').retinanet(num_classes=2, train_generator = generator, depth=args.depth, image_min_side=args.image_min_side, image_max_side= args.image_max_side)
    model.load_weights(args.model, by_name=True, skip_mismatch=True)

    model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001),
        metrics=['accuracy']
    )
    
    # print model summary
    # print(model.summary())

    print ('generating visualization...', args.model)
    layer_name = args.layer
    layer_size = args.layer_size
    print ('layer Name: ', layer_name)

    for image_index in range(10, 40):
        new_image_color = generator.load_image(image_index)
        cv2.imwrite(args.save_path + 'input/'+str(image_index)+'.png', new_image_color)

        new_image_gray = generator.convert_RGB_to_GRAY(new_image_color)
        images = generator.return_output_of_generator(new_image_color, new_image_gray)        
        
        label_model = keras.models.Model(inputs = model.input, outputs = model.get_layer(layer_name).output)
        label_output = label_model.predict(images, verbose = 2)
        label_output = np.rollaxis(np.rollaxis(label_output, 3,1), 0,4)
        for index in range(label_output.shape[0]):
            label = (label_output[index]).reshape(layer_size,layer_size,1)
            label *=255

            label = KImage.array_to_img(label)
            label = np.asarray(label)

            heatmap_img = cv2.applyColorMap(label, cv2.COLORMAP_JET)
            cv2.imwrite(args.save_path +layer_name+'/output/'+str(image_index)+'_'+str(layer_name)+'_'+str(index)+'.png', heatmap_img)


# model3 => (608, 608), 31-200                    => Estimated background
# conv1_relu => (304, 304), 31-40                       => convolution on concatenated history block (background, current frame)
# conv1_relu_new_image => (304,304), 31-40              => convolution on curent colored frame
# pool1 => (152, 152), 31-40                            => convolution and max pooling (detection block) on concatenated history block (background, current frame)
# pool1_new_image => (152, 152), 31-40                  => convolution and max pooling (detection block) on curent colored frame
# P3 => (76,76), 31-40                                  => pyramid layer P3
# P4 => (38, 38), 31-40                                 => pyramid layer P4
# P5 => (19,19), 31-40                                  => pyramid layer P5

if __name__ == '__main__':
    main()

