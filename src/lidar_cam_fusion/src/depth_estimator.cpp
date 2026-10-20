#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <unordered_set>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "vision_interfaces/msg/detection_array.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/time.h"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Vector3.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"

using det = vision_interfaces::msg::DetectionArray;
using cl = sensor_msgs::msg::PointCloud2;
using pol = message_filters::sync_policies::ApproximateTime<det, cl>;
using out = vision_interfaces::msg::DetectionArray;

class Fusion : public rclcpp::Node {
    public: 
    Fusion() : Node("depth_estimator"), tf_buffer(this->get_clock()), tf_listener(tf_buffer){

        //parameters
        auto flat = declare_parameter<std::vector<std::string>>("ipm_classes", std::vector<std::string>{"pothole", "tyre"});
        ipm_classes.insert(flat.begin(), flat.end());
        stop_sign_class      = this->declare_parameter<std::string>("stop_sign_class", "sign");
        stop_line_max_age    = this->declare_parameter<double>("stop_line_max_age", 0.5); 
        stop_line_min_points = this->declare_parameter<int>("stop_line_min_points", 5);
        stop_line_percentile = this->declare_parameter<double>("stop_line_percentile", 0.1); 
        ground_frame  = declare_parameter<std::string>("ground_frame", "link_base");
        ground_z      = declare_parameter<double>("ground_z", 0.0);       
        max_ipm_range = declare_parameter<double>("max_ipm_range", 40.0); 

        //publishers and subscriptions
        camera_subscription = this->create_subscription<sensor_msgs::msg::CameraInfo>("zed_node/monocamera/camera_info", rclcpp::SensorDataQoS(), std::bind(&Fusion::camera_callback, this, std::placeholders::_1));
        cloud_subscription.subscribe(this, "obstacle_cloud", rmw_qos_profile_sensor_data);
        detection_subscription.subscribe(this, "detections", rmw_qos_profile_sensor_data);
        stop_line_subscription = this->create_subscription<cl>("pc_stopline", rclcpp::SensorDataQoS(), std::bind(&Fusion::stop_line_callback, this, std::placeholders::_1));
        sync = std::make_shared<message_filters::Synchronizer<pol>>(pol(10),detection_subscription,cloud_subscription);
        sync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
        sync->registerCallback(&Fusion::fusion_callback, this);
        distance_publisher = this->create_publisher<out>("detections_depth", 10);
    }
    
    private: 
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    bool ok = false;
    std::unordered_set<std::string> ipm_classes;
    std::string ground_frame;
    double ground_z = 0, max_ipm_range = 40;
    std::string stop_sign_class = "stop_sign";
    int stop_line_min_points = 5;
    double stop_line_percentile = 0.1;
    double stop_line_max_age = 0.5;
    rclcpp::Subscription<cl>::SharedPtr stop_line_subscription;
    bool stop_line_valid = false;
    float stop_line_dist = 0.0f;
    size_t stop_line_n = 0;
    rclcpp::Time stop_line_stamp{0, 0 , RCL_ROS_TIME};
        
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

    void stop_line_callback(const cl::SharedPtr cloud) {
        if(ok == false){
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Did not receive camera info");
        return;
    }
    geometry_msgs::msg::TransformStamped tf;

    try {
    tf = tf_buffer.lookupTransform(camera_frame, cloud->header.frame_id, tf2::TimePointZero);
    }
    catch(const tf2::TransformException & e) {
        RCLCPP_WARN(this->get_logger(), "Stop line transform failed: %s", e.what());
        return;
    }

    cl tc;
    tf2::doTransform(*cloud, tc, tf);
    std::vector<float> depths;
    sensor_msgs::PointCloud2ConstIterator<float> ix(tc,"x");
    sensor_msgs::PointCloud2ConstIterator<float> iy(tc,"y");
    sensor_msgs::PointCloud2ConstIterator<float> iz(tc,"z");

    for (; ix != ix.end(); ++ix, ++iy, ++iz) {   
       if (!std::isfinite(*ix) || !std::isfinite(*iy) || !std::isfinite(*iz) || *ix <= 0.1f)
           continue;
       depths.push_back(*ix);
   }

    stop_line_n = depths.size();
    stop_line_valid = depths.size() >= static_cast<size_t>(stop_line_min_points);
    stop_line_stamp = rclcpp::Time(cloud->header.stamp);
    if(stop_line_valid){
        size_t index = static_cast<size_t>(depths.size() * stop_line_percentile);
        std::nth_element(depths.begin(), depths.begin()+index, depths.end());
        stop_line_dist = depths[index];
    }
    }

    bool ipm_depth(double u, double v, const::tf2::Vector3 &origin, const::tf2::Matrix3x3 &R, float &depth) {
        tf2::Vector3 d = R * tf2::Vector3(1.0, -(u - cx) / fx, -(v - cy) / fy);
        if(d.z() > -1e-3)
        return false;
        double t = (ground_z - origin.z())/d.z();
        if (t<=0.0 || t > max_ipm_range)
        return false;
        depth = static_cast<float>(t);
        return true;
    }

        void set_position(vision_interfaces::msg::Detection2D &result, double u, double v, double d) {   
        result.position.x = d;                       
        result.position.y = -(u - cx) * d / fx;      
        result.position.z = -(v - cy) * d / fy;      
    }   


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
   
    bool have_ground = false;
    tf2::Vector3 ipm_origin;
    tf2::Matrix3x3 ipm_R;

    try {
        auto ground = tf_buffer.lookupTransform(ground_frame, camera_frame, tf2::TimePointZero);
        const auto &q = ground.transform.rotation;
        const auto &t = ground.transform.translation;
        ipm_R = tf2::Matrix3x3(tf2::Quaternion(q.x, q.y, q.z, q.w));
        ipm_origin = tf2::Vector3(t.x, t.y, t.z);
        have_ground = true;
    }
    catch(const tf2::TransformException &e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Ground transform failed: %s", e.what()); 
    }

    out output;
    output.header = detections->header;
    output.header.frame_id = camera_frame;

    for (const auto &b : detections->detections) {
        
        auto result = b;

        if(b.is_stop_sign) {   
            result.n_pts = stop_line_n;
            result.depth_valid = false;  
            double age = std::fabs((rclcpp::Time(detections->header.stamp) - stop_line_stamp).seconds());
          if (stop_line_valid && age <= stop_line_max_age) {
                result.distance = stop_line_dist;
                set_position(result, 0.5 * (b.x1 + b.x2), 0.5 * (b.y1 + b.y2), stop_line_dist);   
                result.depth_valid = true;
            }
            output.detections.push_back(result);  
            continue;
        }


        if(ipm_classes.count(b.class_name)) {
            result.n_pts = 0;
            result.depth_valid = false;
            if (have_ground) {
                float depth = 0.0f;
                double u = 0.5 * (b.x1 + b.x2);
                double v = 0.5 * (b.y1 + b.y2);
                if (ipm_depth(u,v,ipm_origin,ipm_R,depth)){
                    result.distance = depth;
                    set_position(result, u, v, depth);
                    result.depth_valid = true;
                }
            }
            output.detections.push_back(result);
            continue;
        }

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

        result.n_pts = depths.size();
        result.depth_valid = depths.size() >= 3;

        if (result.depth_valid) {

            size_t index =
                static_cast<size_t>(depths.size() * 0.25);

            std::nth_element(
                depths.begin(),
                depths.begin() + index,
                depths.end()
            );

            result.distance = depths[index];
            set_position(result, 0.5 * (b.x1 + b.x2), 0.5 * (b.y1 + b.y2), result.distance); 
        }
        output.detections.push_back(result);
    }

    distance_publisher->publish(output);

    RCLCPP_INFO(
        this->get_logger(),
        "Published distances for %zu detections",
        output.detections.size()
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

