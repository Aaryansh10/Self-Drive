# vision_py

## EXECUTABLES:
* vision.py: Vision node which uses pytorch. Publishes the pointclouds for the lane markings, detections of the obstacles and the centerline paths for both the left and right lanes.
* vision_cv.py: Vision node which uses opencv. Publishes the pointclouds for the lane markings, detections of the obstacles and the centerline paths for both the left and right lanes.

### centerline.py:
* Contains the class which can be instantiated to generate the centerline paths for both the left and right lanes.
* The function generate_centerline() is the entry point to the entire process. The function expects the pointclouds of the leading edge of the white lines and the yellow lines as parameters.

#### Process:
* It first splits the points in the point cloud into separate lines, by assigning it a line id. This is done on the basis of the euclidean distances between the points: if the distance is beyond a certain threshold then it marks the point as the starting point of a new line.
* It then bins the 