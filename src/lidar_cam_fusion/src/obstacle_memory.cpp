#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <limits>
#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "fusion_interfaces/msg/detection_distance_array.hpp"
#include "fusion_interfaces/msg/tracked_obstacle.hpp"
#include "fusion_interfaces/msg/tracked_obstacle_array.hpp"

using det = fusion_interfaces::msg::DetectionDistanceArray;
using cl = sensor_msgs::msg::PointCloud2;
using out = fusion_interfaces::msg::TrackedObstacleArray;

struct Track {
    uint32_t id = 0;
    uint32_t class_id = 0;
    std::string class_name;
    Eigen::Vector3d odom;
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
    odom_frame = declare_parameter<std::string>("odom_frame", "camera_init");
    base_frame = declare_parameter<std::string>("base_frame", "link_base");
    gate_base = declare_parameter<double>("gate_base", 0.5);
    gate_range = declare_parameter<double>("gate_range", 0.05);
    pass_margin = declare_parameter<double>("pass_margin", 0.5);
    max_visible_range = declare_parameter<double>("max_visible_range", 0.5);
    confirm_hits = declare_parameter<int>("confirm_hits", 3);
    confirm_window = declare_parameter<int>("confirm_window", 5);
    max_out_of_lane = declare_parameter<int>("max_out_of_lane", 4);
    max_visible_misses = declare_parameter<int>("max_visible_misses", 10);
    
    camera_subscription = this->create_subscription<sensor_msgs::msg::CameraInfo>("zed_node/monocamera/camera_info", rclcpp::SensorDataQoS(), std::bind(&Fusion::camera_callback, this, std::placeholders::_1));
    detection_subscription = this->create_subscription<det>("detected_distance", 10, std::bind(&ObstacleMemory::detection_callback, this, std::placeholders::_1));
    pub = this->create_publisher<out>("tracked_obstacles", 10);
    timer = create_wall_timer(std::chrono::milliseconds(50), std::bind(&Fusion::camera_callback, this, std::placeholders::_1));
   }
    
    
    private:

    rclcpp::Subscription<det>::SharedPtr detection_sub;
    std::vector<Track> tracks;
    static Eigen::Vector3d transform_point(const Eigen::Vector3d &point, const TransformStamped &transform) {
      geometry_msgs::msg::PointStamped point_in, point_out;
      point_in.point.x = point.x();
      point_in.point.y = point.y();
      point_in.point.z = point.z();
      tf2::doTransform(point_in, point_out, transform);
      return Eigen::Vector3d(point_out.point.x, point_out.point.y, point_out.point.z);
    }

    void camera_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
     fx = msg->k[0];
     cx = msg->k[2];
     fy = msg->k[4];
     cy = msg->k[5];
     img_w = msg->width;
     img_h = msg->height;
     ok = true;
     camera_frame = msg->header.frame_id;
     RCLCPP_INFO(this->get_logger(), "Received camera info");
   }

   void detection_callback(const DetectionDistanceArray::SharedPtr msg) {
    TransformStamped odom_from_source;
    if (lookup_transform(odom_frame, msg->header.frame_id, msg->header.stamp, 0.1, odom_from_source) == false){
      return;
    }

    for (auto & track : tracks) {
      track.matched_this_frame = frame;
      track.age_frames++;
    }

    int best_index = -1;
    double best_distance = gate_base + (gate_range * detection.distance);
    for (size_t i = 0; i < tracks.size(); ++i) {
      const Track & candidate = tracks[i];
      if (candidate.matched_this_frame || candidate.class_id != static_cast<int>(detection.class_id))
      continue;
    double distance = (candidate.position_odom - measurement).norm();
    if (distance < best_distance){
      best_distance = distance;
      best_index = static_cast<int>(index);
    }
   }
  }




   } 


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleMemoryNode>());
  rclcpp::shutdown();
  return 0;
}