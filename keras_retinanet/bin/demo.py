#!/usr/bin/env python

"""
MotionRec Demo
"""

import argparse
import cv2
import glob
import os
import sys

import keras
import tensorflow as tf

# For using single gpu, enter gpu id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from .. import models
from ..preprocessing.demo_generator import CSVGenerator
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.demo import evaluate
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
            # args.annotations,
            args.classes,
            video_path = args.video_path,
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
    parser.add_argument('--save-path',        help='Path for saving images with detections.', default='./outputs/test')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=608)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=608)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--depth',      help='image frame depth.', default=10, type=int)
    parser.add_argument('--video-path',    help='video name ', default='./demo_video/sofa.avi')

    return parser.parse_args(args)

def FrameCapture(video_path):
    """ Store image frame to demo_video/test_image_frame/
    """
    vidObj = cv2.VideoCapture(video_path)
    video_folder = "/".join(video_path.split('/')[:-1])+'/'
    file_output_path = video_folder + 'test_image_frame/'
    if not os.path.exists(file_output_path):
        print("folder path not exists for saving video frames, creating folder %s", file_output_path)
        os.makedirs(file_output_path)
    count = 0
    success = 1
    while success: 
        success, image = vidObj.read()
        if(success):
            cv2.imwrite(file_output_path+"in%d.jpg" % count, image) 
            count += 1
        else:
            print('error in image reading or folder ends')
        if(count == 500):
            break

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.video_path is not None and os.path.isfile(args.video_path):
        FrameCapture(args.video_path)
    else:
        print('Error in path of video..', args.video_path)
        exit()
    
    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, anchor_params=anchor_params)

    # print model summary
    print(model.summary())

    # start evaluation
    average_precisions = evaluate(
        generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
