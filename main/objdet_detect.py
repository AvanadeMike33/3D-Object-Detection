# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import torch
from easydict import EasyDict as edict

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing 

from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2


# load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()  

    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))    
    
    # set parameters according to model type
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.min_iou = 0.5
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False

    elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######     
        #######
        print("student task ID_S3_EX1-3")
# Path to the FPN-ResNet object-detection model.
        configs.model_path = os.path.join(
            parent_path,
            'tools',
            'objdet_models',
            'resnet'
        )

        # Path to the pretrained model weights.
        configs.pretrained_filename = os.path.join(
            configs.model_path,
            'pretrained',
            'fpn_resnet_18_epoch_300.pth'
        )

        # Model architecture configuration.
        configs.arch = 'fpn_resnet_18'
        configs.num_layers = 18

        # Object classes:
        # 0 = pedestrian
        # 1 = car
        # 2 = cyclist
        configs.num_classes = 3

        # Number of detections returned by the decoder.
        configs.K = 50

        # Output-stride reduction between the BEV input image
        # and the model output feature map.
        configs.down_ratio = 4

        # Threshold applied to the center heatmap.
        configs.peak_thresh = 0.2

        # Number of convolutional filters in the prediction heads.
        configs.head_conv = 64

        # Prediction heads used by the FPN-ResNet model.
        configs.heads = {
            'hm_cen': configs.num_classes,
            'cen_offset': 2,
            'direction': 2,
            'z_coor': 1,
            'dim': 3
        }

        # Test and output-related parameters.
        configs.save_test_output = False
        configs.output_format = 'image'
        configs.output_video_fn = 'out_fpn_resnet_18'
        configs.output_width = 608
        #######
        ####### ID_S3_EX1-3 END #######     

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = True # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    return configs


# load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()    

    # birds-eye view (bev) parameters
    configs.lim_x = [0, 50] # detection range in m
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0] # reflected lidar intensity
    configs.bev_width = 608  # pixel resolution of bev image
    configs.bev_height = 608 

    # add model-dependent parameters
    configs = load_configs_model(model_name, configs)

    # visualization parameters
    configs.output_width = 608 # width of result image (height may vary)
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]] # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    return configs


# create model according to selected model type
def create_model(configs):

    # check for availability of model file
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

    # create model depending on architecture name
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
    
    elif 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        
        ####### ID_S3_EX1-4 START #######     
        #######
        print("student task ID_S3_EX1-4")
        model = fpn_resnet.get_pose_net(
            num_layers=configs.num_layers,
            heads=configs.heads,
            head_conv=configs.head_conv,
            imagenet_pretrained=False
        )
        #######
        ####### ID_S3_EX1-4 END #######     
    
    else:
        assert False, 'Undefined model backbone'

    # load model weights
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_filename))

    # set model to evaluation state
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)  # load model to either cpu or gpu
    model.eval()          

    return model


# detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs):

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():  

        # perform inference
        outputs = model(input_bev_maps)

        # decode model output into target object format
        if 'darknet' in configs.arch:

            # perform post-processing
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])    

        elif 'fpn_resnet' in configs.arch:
            # decode output and perform post-processing
            
            ####### ID_S3_EX1-5 START #######     
            #######
            print("student task ID_S3_EX1-5")
            outputs['hm_cen'] = torch.sigmoid(
                outputs['hm_cen']
            )

            # Apply the sigmoid activation to the center offset.
            outputs['cen_offset'] = torch.sigmoid(
                outputs['cen_offset']
            )

            # Decode the individual prediction heads into detections.
            detections = decode(
                outputs['hm_cen'],
                outputs['cen_offset'],
                outputs['direction'],
                outputs['z_coor'],
                outputs['dim'],
                K=configs.K
            )

            # Move the detections to the CPU and convert them
            # into a NumPy floating-point array.
            detections = detections.cpu().numpy().astype(
                np.float32
            )

            # Remove detections below the configured confidence threshold
            # and reorganize the results by class.
            detections = post_processing(
                detections,
                configs.num_classes,
                configs.down_ratio,
                configs.peak_thresh
            )

            # This project tracks vehicles. Therefore, select the
            # detections belonging to the car class.
            if (
                detections is not None
                and len(detections) > 0
                and 1 in detections[0]
            ):
                detections = detections[0][1]
            else:
                detections = []
            #######
            ####### ID_S3_EX1-5 END #######     

            

    ####### ID_S3_EX2 START #######     
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = [] 

    ## step 1 : check whether there are any detections
    if detections is not None and len(detections) > 0:

        ## step 2 : loop over all detections
        for obj in detections:

            # The post-processing result of the FPN-ResNet detector
            # contains:
            #
            # [score, x, y, z, h, w, l, yaw]
            #
            # The x and y values are expressed in BEV output-map
            # coordinates and must be converted back into meters.
            if 'fpn_resnet' in configs.arch:
                (
                    score,
                    bev_x,
                    bev_y,
                    z,
                    h,
                    w,
                    l,
                    yaw
                ) = obj

                ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
                x = (
                    bev_y /
                    configs.bev_height *
                    (configs.lim_x[1] - configs.lim_x[0])
                    + configs.lim_x[0]
                )

                y = (
                    bev_x /
                    configs.bev_width *
                    (configs.lim_y[1] - configs.lim_y[0])
                    + configs.lim_y[0]
                )

                z = z + configs.lim_z[0]

                # Keep only objects whose centers lie inside
                # the configured detection limits.
                if (
                    configs.lim_x[0] <= x <= configs.lim_x[1]
                    and configs.lim_y[0] <= y <= configs.lim_y[1]
                    and configs.lim_z[0] <= z <= configs.lim_z[1]
                ):
                    ## step 4 : append the current object to the 'objects' array
                    objects.append(
                        [
                            1,
                            x,
                            y,
                            z,
                            h,
                            w,
                            l,
                            yaw
                        ]
                    )

            elif 'darknet' in configs.arch:
                (
                    obj_class,
                    bev_x,
                    bev_y,
                    z,
                    h,
                    bev_w,
                    bev_l,
                    yaw
                ) = obj

                ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
                x = (
                    bev_y /
                    configs.bev_height *
                    (configs.lim_x[1] - configs.lim_x[0])
                    + configs.lim_x[0]
                )

                y = (
                    bev_x /
                    configs.bev_width *
                    (configs.lim_y[1] - configs.lim_y[0])
                    + configs.lim_y[0]
                )

                z = z + configs.lim_z[0]

                w = (
                    bev_w /
                    configs.bev_width *
                    (configs.lim_y[1] - configs.lim_y[0])
                )

                l = (
                    bev_l /
                    configs.bev_height *
                    (configs.lim_x[1] - configs.lim_x[0])
                )

                if (
                    configs.lim_x[0] <= x <= configs.lim_x[1]
                    and configs.lim_y[0] <= y <= configs.lim_y[1]
                    and configs.lim_z[0] <= z <= configs.lim_z[1]
                ):
                    ## step 4 : append the current object to the 'objects' array
                    objects.append(
                        [
                            obj_class,
                            x,
                            y,
                            z,
                            h,
                            w,
                            l,
                            yaw
                        ]
                    )

    #######
    ####### ID_S3_EX2 END #######

    return objects