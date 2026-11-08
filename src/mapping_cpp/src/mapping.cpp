#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "lane_mapper.hpp"

using PC2 = sensor_msgs::msg::PointCloud2;
using SyncPolicy = message_filters::sync_policies::ExactTime<PC2, PC2>;

class MappingNode : public rclcpp::Node {
    public:
        MappingNode() : Node("mapping_node") {
            // Topics / frames
            const auto white_topic = declare_parameter<std::string>("white_topic", "/pc_whitelane");
            const auto yellow_topic = declare_parameter<std::string>("yellow_topic", "/pc_yellowlane");
            const auto map_topic = declare_parameter<std::string>("white_map_topic", "/lane_map/white");
            const auto yellow_map_topic = declare_parameter<std::string>("yellow_map_topic", "/lane_map/yellow");
            odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
            base_frame_ = declare_parameter<std::string>("base_frame", "link_base");
            publish_every_ = std::max<int>(1, declare_parameter<int>("publish_every", 3));

            // TF wait handling
            // A synced cloud is held until TF for its stamp exists. If TF has not arrived
            // after tf_max_wait_ms, the cloud is either dropped or (optionally) placed using
            // the latest available pose.
            tf_max_wait_ms_ = declare_parameter<int>("tf_max_wait_ms", 700);
            tf_poll_ms_ = declare_parameter<int>("tf_poll_ms", 10);
            max_pending_ = size_t(std::max<int>(1, declare_parameter<int>("max_pending", 20)));
            fallback_to_latest_ = declare_parameter<bool>("fallback_to_latest", false);

            // Occupancy Grid Parameters
            lane_mapper::Params p;
            p.resolution = declare_parameter<double>("resolution", 0.1);
            p.width = declare_parameter<int>("width", 250);
            p.height = declare_parameter<int>("height", 250);
            p.roi_x_min = declare_parameter<double>("roi_x_min", 1.02);
            p.roi_x_max = declare_parameter<double>("roi_x_max", 7.72);
            p.roi_half_y_near = declare_parameter<double>("roi_half_y_near", 1.93);
            p.roi_half_y_far = declare_parameter<double>("roi_half_y_far", 11.02);
            p.sigma_base = declare_parameter<double>("sigma_base", 0.09);
            p.sigma_per_m = declare_parameter<double>("sigma_per_m", 0.0);
            p.sigma_per_m2 = declare_parameter<double>("sigma_per_m2", 0.002);
            p.range_ref = declare_parameter<double>("range_ref", 5.0);
            p.l_hit = declare_parameter<double>("l_hit", 0.9);
            p.l_miss = declare_parameter<double>("l_miss", -0.12);
            p.l_min = declare_parameter<double>("l_min", -2.0);
            p.l_max = declare_parameter<double>("l_max", 5.0);
            p.l_occupied = declare_parameter<double>("l_occ", 1.5);
            p.min_points_for_miss = declare_parameter<int>("min_points_for_miss", 30);  // combined count
            p.speckle_radius  = declare_parameter<int>("speckle_radius", 1);
            p.speckle_support = declare_parameter<int>("speckle_support", 3);
            roi_x_min_ = p.roi_x_min;
            roi_x_max_ = p.roi_x_max;
            if (p.resolution <= 0 || p.width <= 0 || p.height <= 0 ||
                p.roi_x_max <= p.roi_x_min || p.range_ref <= 0 || p.sigma_base <= 0)
                throw std::invalid_argument("Invalid lane map geometry or evidence parameters");
            white_grid_ = std::make_unique<lane_mapper::LaneMapper>(p);
            yellow_grid_ = std::make_unique<lane_mapper::LaneMapper>(p);

            // TF (the listener spins its own thread, so the buffer fills independently of our executor)
            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
            tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

            // Sync clouds
            sub_w_.subscribe(this, white_topic, rmw_qos_profile_sensor_data);
            sub_y_.subscribe(this, yellow_topic, rmw_qos_profile_sensor_data);
            sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), sub_w_, sub_y_);
            sync_->registerCallback(
                std::bind(&MappingNode::cb, this, std::placeholders::_1,
                std::placeholders::_2)
            );

            // Retries pending clouds once TF catches up, even if no new cloud arrives
            drain_timer_ = create_wall_timer(
                std::chrono::milliseconds(std::max<int>(1, tf_poll_ms_)),
                std::bind(&MappingNode::drain, this));

            map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(map_topic, rclcpp::QoS(1).reliable().transient_local());
            yellow_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(yellow_map_topic, rclcpp::QoS(1).reliable().transient_local());
            RCLCPP_INFO(get_logger(), "Mapping node started.");
        }

    private:
        struct Pending {
            PC2::ConstSharedPtr white, yellow;
        };

        std::string odom_frame_, base_frame_;
        int publish_every_ = 3;
        int tf_max_wait_ms_ = 300;
        int tf_poll_ms_ = 10;
        size_t max_pending_ = 20;
        bool fallback_to_latest_ = false;
        unsigned frame_ = 0;
        double roi_x_min_ = 0.0;
        double roi_x_max_ = 0.0;

        std::unique_ptr<lane_mapper::LaneMapper> white_grid_, yellow_grid_;
        std::vector<int8_t> buf_;
        lane_mapper::Window win_{};

        // Only touched from the node's executor (callback + timer). Add a mutex if you
        // switch to a multi-threaded executor with a reentrant callback group.
        std::deque<Pending> pending_;

        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
        message_filters::Subscriber<PC2> sub_w_, sub_y_;
        std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
        rclcpp::TimerBase::SharedPtr drain_timer_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_, yellow_map_pub_;

        void cb(const PC2::ConstSharedPtr& white, const PC2::ConstSharedPtr& yellow) {
            if (white->header.frame_id != base_frame_ || yellow->header.frame_id != base_frame_) {
                RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                    "Lane clouds must be in configured base_frame '%s'", base_frame_.c_str());
                return;
            }
            pending_.push_back({white, yellow});
            while (pending_.size() > max_pending_) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "Pending queue full (%zu), dropping oldest cloud (TF too slow?)", max_pending_);
                pending_.pop_front();
            }
            drain();
        }

        // Process queued clouds in order for as long as TF for their stamp is available.
        void drain() {
            while (!pending_.empty()) {
                const Pending& p = pending_.front();
                const auto stamp = p.white->header.stamp;
                const tf2::TimePoint tp = tf2_ros::fromMsg(stamp);

                std::string err;
                if (tf_buffer_->canTransform(odom_frame_, base_frame_, tp, &err)) {
                    // Pose of the robot at "Image" time and not "now". Inference latency would smear lines otherwise
                    try {
                        const auto tf = tf_buffer_->lookupTransform(odom_frame_, base_frame_, tp);
                        process(p, tf);
                    }
                    catch (const tf2::TransformException& e) {
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF lookup failed: %s", e.what());
                    }
                    pending_.pop_front();
                    continue;
                }

                // TF not available yet. Keep waiting until the cloud is too old to be useful.
                const double age_ms = (now() - rclcpp::Time(stamp, get_clock()->get_clock_type())).seconds() * 1000.0;
                if (age_ms <= tf_max_wait_ms_)
                    break;  // preserve ordering: stop here and retry on next timer tick

                if (fallback_to_latest_) {
                    try {
                        const auto tf = tf_buffer_->lookupTransform(odom_frame_, base_frame_, tf2::TimePointZero);
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                            "TF for stamp not available after %d ms, using latest pose: %s",
                            tf_max_wait_ms_, err.c_str());
                        process(p, tf);
                    }
                    catch (const tf2::TransformException& e) {
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                            "Fallback TF lookup failed: %s", e.what());
                    }
                } else {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                        "Dropping cloud, TF unavailable after %d ms: %s", tf_max_wait_ms_, err.c_str());
                }
                pending_.pop_front();
            }
        }

        void process(const Pending& p, const geometry_msgs::msg::TransformStamped& tf) {
            const auto& white = p.white;
            const auto& yellow = p.yellow;

            const auto& t = tf.transform.translation;
            const auto& q = tf.transform.rotation;
            const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
            const lane_mapper::Pose2 pose{t.x, t.y, yaw};

            auto points = [this](const PC2& cloud) {
                std::vector<lane_mapper::Point2> pts;
                const size_t n = size_t(cloud.width) * cloud.height;
                if (n == 0 || cloud.data.empty()) return pts;
                pts.reserve(n);
                sensor_msgs::PointCloud2ConstIterator<float> ix(cloud, "x"), iy(cloud, "y");
                for (size_t i = 0; i < n; ++i, ++ix, ++iy) {
                    if (std::isfinite(*ix) && std::isfinite(*iy) &&
                        *ix >= roi_x_min_ && *ix <= roi_x_max_) pts.push_back({*ix, *iy});
                }
                return pts;
            };
            const auto white_pts = points(*white), yellow_pts = points(*yellow);
            const int health = static_cast<int>(white_pts.size() + yellow_pts.size());
            white_grid_->insert(white_pts, pose, health);
            yellow_grid_->insert(yellow_pts, pose, health);

            if (++frame_ % publish_every_ != 0)
                return;

            publishMap(*white_grid_, map_pub_, white->header.stamp);
            publishMap(*yellow_grid_, yellow_map_pub_, white->header.stamp);
        }

        void publishMap(const lane_mapper::LaneMapper& grid,
                        const rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr& pub,
                        const builtin_interfaces::msg::Time& stamp) {
            grid.snapshot(buf_, win_);

            nav_msgs::msg::OccupancyGrid msg;
            msg.header.stamp = stamp;
            msg.header.frame_id = odom_frame_;
            msg.info.resolution = float(win_.resolution);
            msg.info.width = uint32_t(win_.width);
            msg.info.height = uint32_t(win_.height);
            msg.info.origin.position.x = win_.origin_x;
            msg.info.origin.position.y = win_.origin_y;
            msg.info.origin.orientation.w = 1.0;
            msg.data = buf_;
            pub->publish(msg);
        }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MappingNode>());
    rclcpp::shutdown();
    return 0;
}