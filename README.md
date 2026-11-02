# 3D-Object-Detection

This project implements a 3D object-detection pipeline using LiDAR data
from the Waymo Open Dataset. The pipeline converts LiDAR range images
into point clouds, creates Bird's-Eye View (BEV) representations,
performs model-based vehicle detection, and evaluates detection
performance using Intersection over Union (IoU), precision, and recall.


## 1. LiDAR Point Cloud from Range Image

### ID_S1_EX1 - Range Image Visualization

The LiDAR range image contains multiple channels. For this project,
the range and intensity channels were extracted and processed.

The range channel represents the distance between the LiDAR sensor and
the reflecting object. Invalid negative values were set to zero and the
remaining values were normalized to an 8-bit range.

The intensity channel represents the strength of the reflected LiDAR
signal. To reduce the influence of extreme outliers, the intensity
values were clipped using the 1st and 99th percentiles before being
mapped to an 8-bit representation.

Only the forward-facing +/-90 degree region was retained.

The processed range and intensity images were then stacked vertically
for visualization with OpenCV.

The resulting visualization therefore contains:

- Range information in the upper half.
- Intensity information in the lower half.
- A 180-degree forward-facing field of view.



## ID_S1_EX2 - Point Cloud Visualization

The LiDAR point cloud was visualized using Open3D.

Each LiDAR return was converted into a three-dimensional point using
its X, Y and Z coordinates. For the analysis presented below, points
were rendered in white against a black background to improve
visibility.

Open3D was used to rotate, translate and zoom the point cloud so that
individual vehicles could be inspected.

## Vehicle Examples

Ten vehicle examples with different degrees of LiDAR visibility were
selected.

### Vehicle 1 - Close Range, High Point Density

<img width="427" height="240" alt="image" src="https://github.com/user-attachments/assets/11d79af3-dc71-4c89-908c-1775b22b396d" />


The vehicle is located relatively close to the LiDAR sensor and
therefore contains a high number of returns. Multiple LiDAR scan lines
cover the vehicle surface, making the overall geometry comparatively
easy to recognize.

### Vehicle 2 - Dense Oblique View

<img width="275" height="235" alt="image" src="https://github.com/user-attachments/assets/40b541e2-2678-4856-872c-0fad38a15257" />


The vehicle is observed from an oblique direction. Several horizontal
scan lines intersect the body and provide information about both the
side and upper surfaces.

### Vehicle 3 - Medium Distance

<img width="263" height="187" alt="image" src="https://github.com/user-attachments/assets/c6355e2a-ae92-4332-90ab-eb42ead63a2b" />


At medium distance, the number of LiDAR returns is lower than for the
closest vehicles. Nevertheless, the compact rectangular geometry of
the vehicle remains distinguishable from the surrounding road.

### Vehicle 4 - Vehicle in Clutter

<img width="1445" height="317" alt="image" src="https://github.com/user-attachments/assets/d87bc98e-b67c-46f5-bb24-da211aab17df" />


This example shows a vehicle close to other structures and LiDAR
returns. Separation from the background is more difficult because
points from surrounding objects occur at similar locations.

### Vehicle 5 - Partial Visibility

<img width="483" height="279" alt="image" src="https://github.com/user-attachments/assets/c0931b0b-bb93-428f-bbb2-ac535b8616d0" />


Only part of the vehicle surface is well represented. This demonstrates
how occlusion and viewing direction can result in incomplete LiDAR
geometry even when the object itself is present.

### Vehicle 6 - Side Profile

<img width="636" height="237" alt="image" src="https://github.com/user-attachments/assets/0cca8b99-9027-4a02-a518-300bed025407" />


The elongated arrangement of LiDAR returns provides evidence of the
vehicle side profile. Several scan lines remain approximately parallel
across the visible body.

### Vehicle 7 - Reduced Point Density

<img width="389" height="287" alt="image" src="https://github.com/user-attachments/assets/63d05385-7826-433e-bae6-de91a3e8ad9d" />


The vehicle contains fewer returns than the close-range examples.
Individual scan lines are more clearly separated and the surface
between those lines is not directly observed.

### Vehicle 8 - Multiple Nearby Objects

<img width="1445" height="317" alt="image" src="https://github.com/user-attachments/assets/e8b24057-3d54-46e6-b994-3c01aedd8772" />


This example contains nearby objects at similar ranges. The point-cloud
representation demonstrates the importance of spatial separation when
identifying individual vehicles.

### Vehicle 9 - Distant Vehicle

<img width="427" height="240" alt="image" src="https://github.com/user-attachments/assets/be81158d-05be-4cbc-9505-cc1d6b0524f8" />


At increased distance, fewer LiDAR beams intersect the vehicle.
Consequently, the object is represented by a substantially sparser set
of points and detailed geometry is lost.

### Vehicle 10 - Very Low Visibility

<img width="275" height="235" alt="image" src="https://github.com/user-attachments/assets/125acaae-9785-4e65-8c7f-98a8fc2372d6" />


This is one of the most challenging examples. Only a relatively small
set of LiDAR returns represents the vehicle. The example illustrates
how object identification becomes increasingly difficult as visibility
and point density decrease.



## Stable Vehicle Features in the LiDAR Point Cloud

Inspection of the ten examples revealed several recurring
characteristics.

### Horizontal Scan-Line Structure

One of the most consistent features is the presence of approximately
parallel LiDAR scan lines across visible vehicle surfaces. The pattern
is particularly clear for nearby vehicles.

### Compact Geometric Structure

Vehicles generally form compact clusters compared with large
continuous surfaces such as roads, walls and buildings.

### Approximately Planar Surfaces

Vehicle roofs, sides, fronts and rear surfaces frequently produce
groups of points that approximate planar structures.

### Height Relative to the Road

Vehicle points occupy a limited vertical interval above the road
surface, which helps distinguish vehicles from both the ground and
taller environmental structures.

### Distance-Dependent Point Density

Point density decreases significantly with distance. Nearby vehicles
may contain many scan lines, while distant vehicles may be represented
by only a small number of returns.

### Occlusion

Occlusion causes characteristic gaps in the point cloud. A vehicle
behind another object may have only its upper surface, rear surface or
one side represented.

Overall, vehicle geometry in LiDAR data is not defined by one
individual feature. Reliable identification instead results from a
combination of spatial extent, height, point density, surface geometry,
and the characteristic LiDAR scan-line pattern.



# 2. Bird's-Eye View from LiDAR Point Cloud

## ID_S2_EX1 - Coordinate Conversion

LiDAR measurements are originally represented in metric vehicle
coordinates.

To construct the Bird's-Eye View, X and Y coordinates were converted
from meters to pixel coordinates according to the configured detection
limits and BEV dimensions.

The X direction represents longitudinal distance from the ego vehicle,
while the Y direction represents lateral displacement.

Only points inside the configured region of interest were retained.



## ID_S2_EX2 - Intensity Layer

For each BEV cell, LiDAR points were sorted by X and Y position and by
height.

When multiple points occupied the same BEV cell, the uppermost point
was retained.

The intensity values were normalized while reducing the effect of
extreme intensity values. This provides better contrast between
vehicles and the surrounding environment.

The resulting intensity layer forms one channel of the BEV input
tensor.



## ID_S2_EX3 - Height Layer

The height layer uses the same sorted and pruned point cloud generated
during intensity processing.

The height of the uppermost LiDAR point in each BEV cell was normalized
according to the configured vertical detection range:

    normalized_height =
        height / (max_height - min_height)

The normalized values were stored in the height channel of the BEV
representation.

A third density channel represents the number of LiDAR points falling
inside each BEV cell using logarithmic normalization.

The final BEV representation therefore consists of three channels:

1. Intensity
2. Height
3. Density



# 3. Model-Based Object Detection

## ID_S3_EX1 - Detection Models

The project supports two model architectures:

- Complex YOLO
- FPN-ResNet / SFA3D-based detection

Complex YOLO directly predicts oriented vehicle bounding boxes from
the BEV representation.

The second model uses a feature-pyramid ResNet architecture and
separate prediction heads for:

- object-center heatmap
- center offset
- object orientation
- vertical position
- object dimensions

The network output is decoded and post-processed before conversion into
the common bounding-box representation.


## ID_S3_EX2 - Extracting 3D Bounding Boxes

Model detections are produced in BEV pixel coordinates.

The detection coordinates and dimensions are converted back into
metric vehicle coordinates using the configured BEV dimensions and
detection limits.

Each resulting object is represented as:

    [class-id, x, y, z, h, w, l, yaw]

where:

- `class-id` identifies the detected object class
- `x, y, z` represent the object center
- `h, w, l` represent height, width and length
- `yaw` represents object orientation



# 4. Object Detection Performance

## ID_S4_EX1 - Intersection over Union

Detection performance was evaluated by comparing predicted bounding
boxes with ground-truth labels.

For every valid ground-truth label and detection pair, the four
Bird's-Eye View corners were generated using
`tools.compute_box_corners`.

The resulting corners were converted to Shapely polygons.

Intersection over Union was calculated as:

    IoU = intersection_area / union_area

Only matches exceeding the configured minimum IoU threshold were
considered valid.

If multiple detections matched the same label, the match with the
highest IoU was retained.



## ID_S4_EX2 - False Positives and False Negatives

After determining the valid object-label matches:

    True Positives = matched valid objects

    False Negatives =
        total valid ground-truth objects - true positives

    False Positives =
        total detections - true positives

These values were accumulated across the evaluated frames.



## ID_S4_EX3 - Precision and Recall

Precision measures the proportion of detections that correspond to
valid objects:

    Precision = TP / (TP + FP)

Recall measures the proportion of ground-truth objects successfully
detected:

    Recall = TP / (TP + FN)

These metrics were calculated across all evaluated frames.


Visual inspection of the LiDAR point cloud shows that vehicles maintain
several characteristic properties, particularly compact geometry,
planar surfaces, limited height above the road and repeated horizontal
scan-line patterns.

The experiments also demonstrate the strong effect of distance and
occlusion on LiDAR point density. Close vehicles provide detailed
geometric information, while distant or partially occluded vehicles
may be represented by only a small number of returns.

These observations motivate the use of BEV representations and
learning-based detection models, which can combine spatial, intensity,
height and density information for robust 3D vehicle detection.
