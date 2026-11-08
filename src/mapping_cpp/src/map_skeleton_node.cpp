#include <memory>
#include <stdexcept>
#include <string>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include "mapping_cpp/skeleton.hpp"

class MapSkeletonNode : public rclcpp::Node {
 public:
  MapSkeletonNode() : Node("map_skeleton_node") {
    threshold_ = declare_parameter<int>("occupied_threshold", 50);
    min_cells_ = declare_parameter<int>("min_component_cells", 3);
    if (threshold_ < 1 || threshold_ > 100 || min_cells_ < 1)
      throw std::invalid_argument("Invalid skeleton threshold or component size");
    const auto qos = rclcpp::QoS(1).reliable().transient_local();
    for (size_t i = 0; i < 2; ++i) {
      const std::string color = i == 0 ? "white" : "yellow";
      const auto input = declare_parameter<std::string>(color + "_map_topic", "/lane_map/" + color);
      const auto output = declare_parameter<std::string>(color + "_skeleton_topic", "/lane_map/" + color + "_skeleton");
      pubs_[i] = create_publisher<Grid>(output, qos);
      subs_[i] = create_subscription<Grid>(input, qos, [this, i](Grid::ConstSharedPtr map) {
        try {
          Grid skeleton = *map;  // preserve stamp, frame, resolution and rolling origin
          skeleton.data = lane_mapper::skeletonize(map->data, map->info.width, map->info.height,
                                                   threshold_, min_cells_);
          pubs_[i]->publish(skeleton);
        } catch (const std::exception& e) {
          RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Cannot skeletonize map: %s", e.what());
        }
      });
    }
  }
 private:
  using Grid = nav_msgs::msg::OccupancyGrid;
  int threshold_, min_cells_;
  rclcpp::Publisher<Grid>::SharedPtr pubs_[2];
  rclcpp::Subscription<Grid>::SharedPtr subs_[2];
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MapSkeletonNode>());
  rclcpp::shutdown();
  return 0;
}
