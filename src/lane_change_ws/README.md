How to run ?

packages to be build : 
self_drive_course
sim_utils_py
steve_description
vision_interfaces vision_py
lane_change 
lane_change_interfaces mapping_cpp

T1 : __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch self_drive_course self_drive_course.launch.py
T2 :__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch steve_description robot_gazebo.launch.py
T3 :ros2 run vision_py vision
T4 :ros2 launch lane_change lane_system.launch.py use_sim_time:=true
T5 :ros2 launch mapping_cpp mapping.launch.py
T6 :ros2 run mapping_cpp path_bridge

Publish lane change action request :
ros2 action send_goal /lane_change lane_change_interfaces/action/LaneChange   "{obstacle_distance: 0.0, completion_distance: 0.0, target_speed: 0.0, target_lane_free: true}"

check :
ros2 topic hz /path/selected_centerline
ros2 topic info /lane_change/selected_lane -v 
ros2 topic info /path/selected_centerline -v



WORKSHOP 
1. lane_change/lane_path_mux.cpp


lane-change server is publishing an Int32 message

RIGHT CENTERLINE  & LEFT CENTERLINE >  LANE PATH MUX    > SELECTED CENTERLINE > Nav2
                                           |
                                      selected_lane              
                                        1 = right
                                        2 = left

Create LanePathMux node

Set:
    RIGHT = 1
    LEFT  = 2

Read parameters:
    right_centerline_topic
    left_centerline_topic
    selected_centerline_topic
    selected_lane_topic
    initial_lane

If initial_lane == "left":
    selected_lane = LEFT
Else:
    selected_lane = RIGHT

Subscribe to:
    right_centerline_topic
    left_centerline_topic
    selected_lane_topic

Create publisher:
    selected_centerline_topic

WHEN a path arrives: (on_path)
    (Determine which lane it belongs to)
    If path is empty:
        ignore it
        return
    Store this path as the latest path for that lane
    If this lane == currently selected lane:
        publish this path to selected_centerline
    Else:
        do not publish it

WHEN selected_lane message arrives:                            
    If lane is neither RIGHT nor LEFT:
        print warning
        ignore it
        return
    If new lane != current lane:
        print:
            RIGHT → LEFT
    selected_lane = new lane
    Check whether we already have a path for new lane
    If path exists:
        publish it immediately
    Else:
        print warning:
            "No LEFT centreline received yet"


2. lane_change/lane_change_server.cpp

 creates the server for lane change action
 subscribes to /left_centerline , /right_centerline 
 publishes /selected_lane

 Right = 1 , left = 2

Set initial lane
    IF initial_lane is "left"
        current_lane = LEFT (2)
    ELSE
        current_lane = RIGHT (1)

Publish current_lane to /lane_change/selected_lane

LaneChange action server
  When a goal is received:
    Accept the goal
  When cancellation is requested:
    Accept cancellation
  When goal is executed:
     IF current_lane is RIGHT
        current_lane = LEFT
     ELSE
        current_lane = RIGHT

Publish current_lane on /lane_change/selected_lane

Mark action as SUCCEEDED

Keep server running and wait for requests


PENDING : 

1. has to trigerred btw ( 13 to 10 ft )
2. should complete the whole transition btw 10-13 ft 






