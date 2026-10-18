#include <rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include 

class Mapper : public rclcpp::Node {
    public:
        Mapper() : Node("mapper_node") {
            
        }
    
    private:
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::ConstSharedPtr pc_sub_;
        
};