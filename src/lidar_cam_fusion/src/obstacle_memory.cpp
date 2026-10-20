#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <limits>
#include <chrono>
#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "tf2/time.h"
#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "vision_interfaces/msg/detection_array.hpp"
#include "fusion_interfaces/msg/tracked_obstacle.hpp"
#include "fusion_interfaces/msg/tracked_obstacle_array.hpp"

using det = vision_interfaces::msg::DetectionDistanceArray;
using trmsg = fusion_interfaces::msg::TrackedObstacle;
using TransformStamped = geometry_msgs::msg::TransformStamped;
using out = fusion_interfaces::msg::TrackedObstacleArray;
using TimeMsg = builtin_interfaces::msg::Time;

struct Track {
    uint32_t id = 0;
    int32_t class_id = 0;
    std::string class_name;
    Eigen::Vector3d position_odom;
    int hits = 1;
    int age = 1;
    int misses = 0;
    int consecutive_out_of_lane = 0;
    bool confirmed = false;
    bool matched = false;
};

class ObstacleMemory : public rclcpp::Node {
   public:
   ObstacleMemory() : Node("obstacle_memory"), tf_buffer(this->get_clock()), tf_listener(tf_buffer){
    
    //parameters
    odom_frame = declare_parameter<std::string>("odom_frame", "odom");
    base_frame = declare_parameter<std::string>("base_frame", "link_base");
    gate_base = declare_parameter<double>("gate_base", 0.5);
    gate_range = declare_parameter<double>("gate_range", 0.05);
    pass_margin = declare_parameter<double>("pass_margin", 0.5);
    max_visible_range = declare_parameter<double>("max_visible_range", 12.0);
    confirm_hits = declare_parameter<int>("confirm_hits", 3);
    confirm_window = declare_parameter<int>("confirm_window", 5);
    max_out_of_lane = declare_parameter<int>("max_out_of_lane", 4);
    max_visible_misses = declare_parameter<int>("max_visible_misses", 10);
    
    //publishers and subscribers
    camera_subscription = this->create_subscription<sensor_msgs::msg::CameraInfo>("zed_node/monocamera/camera_info", rclcpp::SensorDataQoS(), std::bind(&ObstacleMemory::camera_callback, this, std::placeholders::_1));
    detection_subscription = this->create_subscription<det>("detections_lane", 10, std::bind(&ObstacleMemory::detection_callback, this, std::placeholders::_1));
    pub = this->create_publisher<out>("tracked_obstacles", 10);
    timer = create_wall_timer(std::chrono::milliseconds(50), std::bind(&ObstacleMemory::timer_callback, this));
   }
    
    
    private:

    std::string odom_frame, base_frame, camera_frame;
    double gate_base, gate_range, pass_margin, max_visible_range;
    double fx=0, fy=0, cx=0, cy=0;
    int confirm_hits, confirm_window, max_out_of_lane, max_visible_misses;
    uint32_t next_id = 1, image_height = 0, image_width = 0;
    bool has_camera_info = false;
    rclcpp::Subscription<det>::SharedPtr detection_subscription;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_subscription;
    rclcpp::Publisher<out>::SharedPtr pub;
    rclcpp::TimerBase::SharedPtr timer;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    std::vector<Track> tracks;

    // Transform a 3D point into another coordinate frame
    static Eigen::Vector3d transform_point(const Eigen::Vector3d &point, const TransformStamped &transform) {
      geometry_msgs::msg::PointStamped point_in, point_out;
      point_in.point.x = point.x();
      point_in.point.y = point.y();
      point_in.point.z = point.z();
      tf2::doTransform(point_in, point_out, transform);
      return Eigen::Vector3d(point_out.point.x, point_out.point.y, point_out.point.z);
    }

    // Check whether obstacle lies inside the camera's visible region
    bool is_visible(const Eigen::Vector3d &position_odom, const TransformStamped &camera_from_odom) const {
      Eigen::Vector3d in_camera = transform_point(position_odom, camera_from_odom);
    // Convert camera coordinates to optical coordinates  
      double x_optical = -in_camera.y();
      double y_optical = -in_camera.z();
      double z_optical =  in_camera.x();
      if (z_optical <= 0.2 || z_optical > max_visible_range) // Reject points too close or beyond the camera
      return false;
       // Project the 3D point onto the 2D image
      double u = (fx * x_optical / z_optical) + cx;
      double v = (fy * y_optical / z_optical) + cy;
      return u >=0 && u < image_width && v >= 0 && v < image_height;
    }

    // Function for looking up the transform between two frames 
    bool lookup_transform(const std::string &target_frame, const std::string &source_frame, const TimeMsg &stamp, double timeout, TransformStamped &result) {
      try {
        result = tf_buffer.lookupTransform(target_frame, source_frame, tf2_ros::fromMsg(stamp), tf2::durationFromSec(timeout));
        return true;
      }
      catch(const tf2::TransformException & e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Lookup transform failed");
        return false;
      }
    }

    void camera_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
     fx = msg->k[0];
     cx = msg->k[2];
     fy = msg->k[4];
     cy = msg->k[5];
     image_width = msg->width;
     image_height = msg->height;
     has_camera_info = true;
     camera_frame = msg->header.frame_id;
     RCLCPP_INFO(this->get_logger(), "Received camera info");
   }

   void detection_callback(const det::SharedPtr msg) {
    
    //Transform detections into odom frame
    TransformStamped odom_from_source;
    if (lookup_transform(odom_frame, msg->header.frame_id, msg->header.stamp, 0.1, odom_from_source) == false){
      return;
    }

    bool any_invalid = false;

    for (auto & track : tracks) {
      track.matched = false;
      track.age++;
    }

    for (const auto &detection : msg->detections) {
      if (detection.is_stop_sign) continue;               
      if(!detection.depth_valid) { 
        any_invalid = true;
        continue;
      }

      Eigen::Vector3d measurement = transform_point(Eigen::Vector3d(detection.position.x, detection.position.y, detection.position.z), odom_from_source);

      int best_index = -1;
      double best_distance = gate_base + (gate_range * detection.distance);
      // Find the nearest unmatched track of the same class
      for (size_t i = 0; i < tracks.size(); ++i) {
        const Track & candidate = tracks[i];
        if (candidate.matched || candidate.class_id != detection.class_id)
        continue;
        double distance = (candidate.position_odom - measurement).norm();
        if (distance < best_distance){
          best_distance = distance;
          best_index = i;
        }
      }

      // Update an existing track if a suitable match was found
      if (best_index >= 0) {
        Track &track = tracks[best_index];
      // Smooth the position estimate
        double gain = std::max(1.0/(track.hits + 1.0) , 0.2);
        track.position_odom += gain * (measurement - track.position_odom);
        track.hits++;
        track.misses = 0;
        track.matched = true;
        track.consecutive_out_of_lane = detection.is_in_lane ? 0 : track.consecutive_out_of_lane + 1;

        if (track.confirmed == false && track.hits >= confirm_hits && track.age <= confirm_window) {
          track.confirmed = true;
          track.id = next_id++;
        }
      }
      // Create a new track for an unmatched detection inside the lane
      else if(detection.is_in_lane) {
        Track track;
        track.class_id = detection.class_id;
        track.class_name = detection.class_name;
        track.position_odom = measurement;
        track.matched = true;
        tracks.push_back(track);
      }
    }

    if(any_invalid == false && has_camera_info) {
      TransformStamped camera_from_odom;
      if(lookup_transform(camera_frame, odom_frame, msg->header.stamp, 0.05, camera_from_odom)) {
        for (auto &track : tracks) {
          if (track.matched == false && is_visible(track.position_odom, camera_from_odom))
          track.misses++;
        }
      }
    }
   }

   void timer_callback() {

    // Get transformation from odom to base_link

    TransformStamped base_from_odom;

    try {
      base_from_odom = tf_buffer.lookupTransform(base_frame, odom_frame, tf2::TimePointZero);
    }
    catch(const tf2::TransformException & e) {
      return;
    }

    // Create output message 

    out output;
    output.header.stamp = this->now();
    output.header.frame_id = base_frame;
    output.nearest_in_lane_id = 0;
    double nearest_forward = std::numeric_limits<double>::max();

    // Loop through all tracks and transform current obstacle into the base frame, check the 4 expiry conditions

    for(auto i = tracks.begin(); i != tracks.end();) {
      Eigen::Vector3d in_base = transform_point(i->position_odom, base_from_odom);

      bool passed = in_base.x() < -pass_margin ;
      bool out_of_lane = i->consecutive_out_of_lane >= max_out_of_lane;
      bool stale = i->misses >= max_visible_misses;
      bool tentative_expired = i->confirmed == false && ((i->age - i->hits) >= 3 || i->age > confirm_window);

    // Delete expired tracks

    if (passed || out_of_lane || stale || tentative_expired) {
      i = tracks.erase(i);
      continue;
    }

    // Add confirmed tracks to output

    if (i->confirmed) {
      trmsg obstacle;

      obstacle.id = i->id;
      obstacle.class_id = i->class_id;
      obstacle.class_name = i->class_name;
      obstacle.position_base.x = in_base.x();
      obstacle.position_base.y = in_base.y();
      obstacle.position_base.z = in_base.z();
      obstacle.position_odom.x = i->position_odom.x();
      obstacle.position_odom.y = i->position_odom.y();
      obstacle.position_odom.z = i->position_odom.z();
      obstacle.range = std::hypot(in_base.x(), in_base.y());
      obstacle.hits = i->hits;
      output.obstacles.push_back(obstacle);

      if (in_base.x() > 0.0 && in_base.x() < nearest_forward){
        nearest_forward = in_base.x();
        output.nearest_in_lane_id = i->id;
      }
    }
    ++i;
  }
    // Publish output

    pub->publish(output);
   }
  };

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleMemory>());
  rclcpp::shutdown();
  return 0;
}