# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import open3d as o3d

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(
        os.path.join(
            os.getcwd(),
            os.path.expanduser(__file__)
        )
    )
)

sys.path.append(
    os.path.normpath(
        os.path.join(
            SCRIPT_DIR,
            PACKAGE_PARENT
        )
    )
)

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import (
    utils as waymo_utils
)
from tools.waymo_reader.simple_waymo_open_dataset_reader import (
    dataset_pb2,
    label_pb2
)

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window(
        window_name='LiDAR Point Cloud',
        width=1280,
        height=720,
        left=50,
        top=50
    )

    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcd.points = o3d.utility.Vector3dVector(
        pcl[:, 0:3]
    )

    # Display all LiDAR points in white so that they remain clearly
    # visible against the black Open3D background.
    colors = np.ones(
        (pcl.shape[0], 3),
        dtype=np.float64
    )

    pcd.colors = o3d.utility.Vector3dVector(
        colors
    )

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    vis.add_geometry(
        pcd
    )

    # Set visualization parameters.
    render_options = vis.get_render_option()

    render_options.background_color = np.asarray(
        [0.0, 0.0, 0.0]
    )

    render_options.point_size = 3.0

    # Reset the viewpoint so Open3D fits the complete point cloud
    # inside the initial view.
    vis.reset_view_point(
        True
    )

    # Callback used to close the window when the right-arrow key is pressed.
    def close_window(vis_instance):
        vis_instance.close()
        return False

    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    vis.register_key_callback(
        262,
        close_window
    )

    vis.run()
    vis.destroy_window()

    #######
    ####### ID_S1_EX2 END #######


# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    lidar = tools.get(
        frame.lasers,
        lidar_name
    )

    range_image, camera_projection, range_image_pose = (
        waymo_utils.parse_range_image_and_camera_projection(
            lidar
        )
    )

    # Convert the Waymo MatrixFloat representation into a NumPy array.
    if (
        hasattr(range_image, 'data')
        and hasattr(range_image, 'shape')
    ):
        ri = np.asarray(
            range_image.data
        ).reshape(
            range_image.shape.dims
        )
    else:
        ri = np.asarray(
            range_image
        )

    # step 2 : extract the range and the intensity channel from the range image
    range_channel = ri[:, :, 0].copy()
    intensity_channel = ri[:, :, 1].copy()

    # step 3 : set values <0 to zero
    range_channel[
        range_channel < 0
    ] = 0

    intensity_channel[
        intensity_channel < 0
    ] = 0

    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    range_min = np.min(
        range_channel
    )

    range_max = np.max(
        range_channel
    )

    if range_max > range_min:
        range_channel = (
            (range_channel - range_min)
            / (range_max - range_min)
            * 255.0
        )
    else:
        range_channel = np.zeros_like(
            range_channel
        )

    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    intensity_p1 = np.percentile(
        intensity_channel,
        1
    )

    intensity_p99 = np.percentile(
        intensity_channel,
        99
    )

    intensity_channel = np.clip(
        intensity_channel,
        intensity_p1,
        intensity_p99
    )

    if intensity_p99 > intensity_p1:
        intensity_channel = (
            (intensity_channel - intensity_p1)
            / (intensity_p99 - intensity_p1)
            * 255.0
        )
    else:
        intensity_channel = np.zeros_like(
            intensity_channel
        )

    # Crop the range and intensity channels to +/- 90 degrees
    # around the forward-facing x-axis.
    img_width = range_channel.shape[1]

    crop_start = img_width // 4
    crop_end = 3 * img_width // 4

    range_channel = range_channel[
        :,
        crop_start:crop_end
    ]

    intensity_channel = intensity_channel[
        :,
        crop_start:crop_end
    ]

    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    img_range_intensity = np.vstack(
        (
            range_channel,
            intensity_channel
        )
    ).astype(
        np.uint8
    )

    #######
    ####### ID_S1_EX1 END #######

    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where(
        (lidar_pcl[:, 0] >= configs.lim_x[0])
        & (lidar_pcl[:, 0] <= configs.lim_x[1])
        & (lidar_pcl[:, 1] >= configs.lim_y[0])
        & (lidar_pcl[:, 1] <= configs.lim_y[1])
        & (lidar_pcl[:, 2] >= configs.lim_z[0])
        & (lidar_pcl[:, 2] <= configs.lim_z[1])
    )

    lidar_pcl = lidar_pcl[
        mask
    ]

    # Return an empty BEV tensor if the filtered point cloud is empty.
    if lidar_pcl.shape[0] == 0:
        bev_maps = np.zeros(
            (
                1,
                3,
                configs.bev_height,
                configs.bev_width
            ),
            dtype=np.float32
        )

        return torch.from_numpy(
            bev_maps
        ).to(
            configs.device,
            non_blocking=True
        ).float()

    # Make a copy so that the original point cloud is not modified.
    lidar_pcl = np.copy(
        lidar_pcl
    )

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = (
        lidar_pcl[:, 2]
        - configs.lim_z[0]
    )

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    bev_discretization = (
        configs.lim_x[1]
        - configs.lim_x[0]
    ) / configs.bev_height

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    lidar_pcl_cpy = np.copy(
        lidar_pcl
    )

    lidar_pcl_cpy[:, 0] = np.floor(
        (
            lidar_pcl_cpy[:, 0]
            - configs.lim_x[0]
        )
        / bev_discretization
    )

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    bev_discretization_y = (
        configs.lim_y[1]
        - configs.lim_y[0]
    ) / configs.bev_width

    lidar_pcl_cpy[:, 1] = np.floor(
        (
            lidar_pcl_cpy[:, 1]
            - configs.lim_y[0]
        )
        / bev_discretization_y
    )

    lidar_pcl_cpy[:, 0] = np.clip(
        lidar_pcl_cpy[:, 0],
        0,
        configs.bev_height
    )

    lidar_pcl_cpy[:, 1] = np.clip(
        lidar_pcl_cpy[:, 1],
        0,
        configs.bev_width
    )

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    # Uncomment the following line for debugging:
    #
    # show_pcl(lidar_pcl)

    #######
    ####### ID_S2_EX1 END #######

    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros(
        (
            configs.bev_height + 1,
            configs.bev_width + 1
        ),
        dtype=np.float32
    )

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    indices = np.lexsort(
        (
            -lidar_pcl_cpy[:, 2],
            lidar_pcl_cpy[:, 1],
            lidar_pcl_cpy[:, 0]
        )
    )

    lidar_pcl_cpy = lidar_pcl_cpy[
        indices
    ]

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, unique_indices, counts = np.unique(
        lidar_pcl_cpy[:, 0:2],
        axis=0,
        return_index=True,
        return_counts=True
    )

    lidar_pcl_top = lidar_pcl_cpy[
        unique_indices
    ]

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    intensity_values = lidar_pcl_top[
        :,
        3
    ].copy()

    intensity_min = np.percentile(
        intensity_values,
        1
    )

    intensity_max = np.percentile(
        intensity_values,
        99
    )

    intensity_values = np.clip(
        intensity_values,
        intensity_min,
        intensity_max
    )

    if intensity_max > intensity_min:
        normalized_intensity = (
            (intensity_values - intensity_min)
            / (intensity_max - intensity_min)
        )
    else:
        normalized_intensity = np.zeros_like(
            intensity_values,
            dtype=np.float32
        )

    intensity_map[
        np.int_(lidar_pcl_top[:, 0]),
        np.int_(lidar_pcl_top[:, 1])
    ] = normalized_intensity

    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    # Uncomment the following lines for debugging and visualization:
    #
    # img_intensity = intensity_map * 255
    # img_intensity = img_intensity.astype(np.uint8)
    # cv2.imshow("Intensity map", img_intensity)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #######
    ####### ID_S2_EX2 END #######

    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros(
        (
            configs.bev_height + 1,
            configs.bev_width + 1
        ),
        dtype=np.float32
    )

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    height_range = (
        configs.lim_z[1]
        - configs.lim_z[0]
    )

    if height_range > 0:
        normalized_height = (
            lidar_pcl_top[:, 2]
            / height_range
        )
    else:
        normalized_height = np.zeros(
            lidar_pcl_top.shape[0],
            dtype=np.float32
        )

    normalized_height = np.clip(
        normalized_height,
        0.0,
        1.0
    )

    height_map[
        np.int_(lidar_pcl_top[:, 0]),
        np.int_(lidar_pcl_top[:, 1])
    ] = normalized_height

    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    # Uncomment the following lines for debugging and visualization:
    #
    # img_height = height_map * 255
    # img_height = img_height.astype(np.uint8)
    # cv2.imshow("Height map", img_height)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #######
    ####### ID_S2_EX3 END #######

    # Compute density layer of the BEV map
    density_map = np.zeros(
        (
            configs.bev_height + 1,
            configs.bev_width + 1
        ),
        dtype=np.float32
    )

    normalized_counts = np.minimum(
        1.0,
        np.log(
            counts + 1
        )
        / np.log(64)
    )

    density_map[
        np.int_(lidar_pcl_top[:, 0]),
        np.int_(lidar_pcl_top[:, 1])
    ] = normalized_counts

    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros(
        (
            3,
            configs.bev_height,
            configs.bev_width
        ),
        dtype=np.float32
    )

    bev_map[2, :, :] = density_map[
        :configs.bev_height,
        :configs.bev_width
    ]  # r_map

    bev_map[1, :, :] = height_map[
        :configs.bev_height,
        :configs.bev_width
    ]  # g_map

    bev_map[0, :, :] = intensity_map[
        :configs.bev_height,
        :configs.bev_width
    ]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape

    bev_maps = np.zeros(
        (
            1,
            s1,
            s2,
            s3
        ),
        dtype=np.float32
    )

    bev_maps[0] = bev_map

    # create tensor from birds-eye view
    bev_maps = torch.from_numpy(
        bev_maps
    )

    input_bev_maps = bev_maps.to(
        configs.device,
        non_blocking=True
    ).float()

    return input_bev_maps