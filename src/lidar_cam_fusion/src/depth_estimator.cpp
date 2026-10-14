#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "vision_interfaces/msg/detection_array.hpp"
#include "fusion_interfaces/msg/detection_distance.hpp"
#include "fusion_interfaces/msg/detection_distance_array.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/time.h"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"

using det = vision_interfaces::msg::DetectionArray;
using cl = sensor_msgs::msg::PointCloud2;
using pol = message_filters::sync_policies::ApproximateTime<det, cl>;
using out = fusion_interfaces::msg::DetectionDistanceArray;

class Fusion : public rclcpp::Node {
    public:
    Fusion() : Node("depth_estimator"), tf_buffer(this->get_clock()), tf_listener(tf_buffer){
        camera_subscription = this->create_subscription<sensor_msgs::msg::CameraInfo>("camera_info", rclcpp::SensorDataQoS(), std::bind(&Fusion::camera_callback, this, std::placeholders::_1));
        cloud_subscription.subscribe(this, "cloud_info", rmw_qos_profile_sensor_data);
        detection_subscription.subscribe(this, "detections", rmw_qos_profile_sensor_data);
        sync = std::make_shared<message_filters::Synchronizer<pol>>(pol(10),detection_subscription,cloud_subscription);
        sync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
        sync->registerCallback(&Fusion::fusion_callback, this);
        distance_publisher = this->create_publisher<out>("detected_distance", 10);
    }
    
    private:      
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    bool ok = false;
        
rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_subscription;
message_filters::Subscriber<det> detection_subscription;
message_filters::Subscriber<cl> cloud_subscription;
std::shared_ptr<message_filters::Synchronizer<pol>> sync;
rclcpp::Publisher<out>::SharedPtr distance_publisher;
tf2_ros::Buffer tf_buffer;
tf2_ros::TransformListener tf_listener;
std::string camera_frame;

   void camera_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
     fx = msg->k[0];
     cx = msg->k[2];
     fy = msg->k[4];
     cy = msg->k[5];
     ok = true;
     camera_frame = msg->header.frame_id;
     RCLCPP_INFO(this->get_logger(), "Received camera info");
   }

    /*void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
        RCLCPP_INFO(this->get_logger(),"Received a LiDAR point cloud");
        }

    void detection_callback(const vision_interfaces::msg::DetectionArray::ConstSharedPtr msg){
        RCLCPP_INFO(this->get_logger(),"Received object detections",msg->detections.size);

    }*/

    void fusion_callback(const det::ConstSharedPtr &detections, const cl::ConstSharedPtr &cloud)
{
    if (ok == false) {
        RCLCPP_WARN(this->get_logger(), "Did not receive camera info");
        return;
    }

    geometry_msgs::msg::TransformStamped transform;

    try {
        transform = tf_buffer.lookupTransform(camera_frame, cloud->header.frame_id, tf2::TimePointZero);
    }
    catch (const tf2::TransformException &e) {
        RCLCPP_WARN(this->get_logger(), "Transform failed: %s", e.what());
        return;
    }

    cl transformed_cloud;
    tf2::doTransform(*cloud, transformed_cloud, transform);

    out output;
    output.header = detections->header;

    for (const auto &b : detections->detections) {
        std::vector<float> depths;

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(
            transformed_cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(
            transformed_cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(
            transformed_cloud, "z");

        while (iter_x != iter_x.end()) {

            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z <= 0.1f)
            {
                ++iter_x;
                ++iter_y;
                ++iter_z;
                continue;
            }

            float u = (fx * x / z) + cx;
            float v = (fy * y / z) + cy;

            if (u >= b.x1 && u <= b.x2 &&
                v >= b.y1 && v <= b.y2)
            {
                depths.push_back(z);
            }
            ++iter_x;
            ++iter_y;
            ++iter_z;
        }

        fusion_interfaces::msg::DetectionDistance result;

        result.class_id = b.class_id;
        result.class_name = b.class_name;
        result.score = b.score;
        result.n_pts = depths.size();
        result.valid = depths.size() >= 3;

        if (result.valid) {

            size_t index =
                static_cast<size_t>(depths.size() * 0.25);

            std::nth_element(
                depths.begin(),
                depths.begin() + index,
                depths.end()
            );

            result.distance = depths[index];
        }
        output.items.push_back(result);
    }

    distance_publisher->publish(output);

    RCLCPP_INFO(
        this->get_logger(),
        "Published distances for %zu detections",
        output.items.size()
    );
}

};

    int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Fusion>());
    rclcpp::shutdown();

    return 0;
}

