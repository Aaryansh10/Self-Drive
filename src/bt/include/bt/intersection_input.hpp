#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include "bt/shared_state.hpp"
#include "bt/stop_line_geometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "vision_interfaces/msg/detection_array.hpp"

namespace bt {
// Validate a transverse stop-line cloud for every intersection decision.
// A recent sign supplies distance; without one, use the validated cloud distance.
class IntersectionInput {
public:
    IntersectionInput(rclcpp::Node::SharedPtr node, std::shared_ptr<SharedState> state)
        : node_(node), state_(std::move(state)), buffer_(node->get_clock()), listener_(buffer_) {
        base_ = node->declare_parameter<std::string>("intersection_base_frame", "link_base");
        odom_ = node->declare_parameter<std::string>("intersection_odom_frame", "odom");
        max_age_ = node->declare_parameter<double>("intersection_input_max_age", 0.5);
        tf_tolerance_ = node->declare_parameter<double>("intersection_tf_tolerance", 0.1);
        sign_ttl_ = node->get_parameter("stop_sign_ttl").as_double();
        min_score_ = node->get_parameter("min_score").as_double();
        real_threshold_ = node->declare_parameter<double>("sign_real_threshold", 0.5);
        min_points_ = node->declare_parameter<int>("stop_line_min_points", 10);
        half_width_ = node->declare_parameter<double>("stop_line_half_width", 1.2);
        min_span_ = node->declare_parameter<double>("stop_line_min_span", 0.6);
        line_tolerance_ = node->declare_parameter<double>("stop_line_line_tolerance", 0.15);
        max_slope_ = node->declare_parameter<double>("stop_line_max_slope", 0.5);
        max_gap_ = node->declare_parameter<double>("stop_line_max_gap", 0.35);
        sign_distance_tolerance_ = node->declare_parameter<double>("stop_line_sign_distance_tolerance", 1.5);
        const auto line_topic = node->declare_parameter<std::string>("stop_line_topic", "/pc_stopline");
        sign_class_ = node->declare_parameter<std::string>("stop_sign_class", "sign");
        const auto sign_topic = node->declare_parameter<std::string>("sign_detections_topic", "/detections_depth");
        signs_ = node->create_subscription<vision_interfaces::msg::DetectionArray>(
            sign_topic, rclcpp::SensorDataQoS(),
            [this](vision_interfaces::msg::DetectionArray::ConstSharedPtr msg) { on_signs(*msg); });
        lines_ = node->create_subscription<sensor_msgs::msg::PointCloud2>(
            line_topic, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { on_line(*msg); });
    }

private:
    bool fresh(double stamp) const {
        const double age = state_->now() - stamp;
        return age >= -0.05 && age <= max_age_;
    }

    Detection line_detection(double distance, int points, const std_msgs::msg::Header& header) {
        geometry_msgs::msg::PointStamped local, world;
        local.header = header;
        local.point.x = distance;
        // This is an anchor on the source frame's forward axis for handled-line
        // memory, not the sign's image-ray position. Do not recompute distance.
        const rclcpp::Time stamp(header.stamp);
        geometry_msgs::msg::TransformStamped transform;
        try {
            transform = buffer_.lookupTransform(
                odom_, header.frame_id, stamp, rclcpp::Duration::from_seconds(0.05));
        } catch (const tf2::ExtrapolationException&) {
            if (!(tf_tolerance_ > 0.0)) throw;
            // At startup the first buffered TF can be a few milliseconds newer
            // than the image. Only approximate the memory anchor, within a
            // bounded skew; the measured detection.distance stays unchanged.
            transform = buffer_.lookupTransform(
                odom_, header.frame_id, rclcpp::Time(0, 0, RCL_ROS_TIME));
            const double tf_time = rclcpp::Time(transform.header.stamp).seconds();
            if (std::abs(tf_time - stamp.seconds()) > tf_tolerance_ ||
                std::abs(state_->now() - tf_time) > max_age_) throw;
            RCLCPP_DEBUG_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                "Using nearby TF for stop-line memory anchor (skew %.3f s)", tf_time - stamp.seconds());
        }
        tf2::doTransform(local, world, transform);
        Detection line;
        line.cls = "stopline";
        line.valid = true;
        line.score = 1.0;
        line.n_pts = points;
        line.distance = distance;
        line.x = world.point.x;
        line.y = world.point.y;
        return line;
    }

    void add_real_sign(std::vector<Detection>& dets) {
        if (!dets.empty() && state_->now() - real_sign_stamp_ <= sign_ttl_ &&
            real_sign_stamp_ > state_->last_intersection_completed()) {
            auto sign = dets.front();
            sign.cls = "sign";
            sign.observed_at = real_sign_stamp_;
            dets.push_back(sign);
        }
    }

    void on_signs(const vision_interfaces::msg::DetectionArray& msg) {
        const double stamp = rclcpp::Time(msg.header.stamp).seconds();
        if (!fresh(stamp)) return;
        const vision_interfaces::msg::Detection2D* nearest = nullptr;
        bool saw_sign = false;
        for (const auto& d : msg.detections) {
            // is_stop_sign may be false for a fake/unknown sign. Its sign class
            // still identifies a detection whose distance was supplied by fusion.
            if (d.class_name != sign_class_ && !d.is_stop_sign) continue;
            if (!std::isfinite(d.score) || d.score < min_score_) continue;
            saw_sign = true;
            if (!d.depth_valid || !std::isfinite(d.distance) || d.distance < 0.0 ||
                d.n_pts < static_cast<unsigned>(std::max(1, min_points_))) continue;
            if (!nearest || d.distance < nearest->distance) nearest = &d;
        }
        if (!saw_sign) return;
        sign_input_stamp_ = stamp;
        sign_valid_ = nearest != nullptr;
        if (nearest) {
            sign_distance_ = nearest->distance;
            sign_points_ = static_cast<int>(nearest->n_pts);
            sign_header_ = msg.header;
            real_sign_stamp_ = -1e9;
            if (nearest->is_stop_sign && std::isfinite(nearest->sign_real_prob) &&
                nearest->sign_real_prob >= real_threshold_ && nearest->sign_real_prob <= 1.0)
                real_sign_stamp_ = stamp;
        }
        // Decisions advance once per cloud observation, not once per message
        // from both topics, so confirmation requires independent cloud frames.
    }

    void update_decision() {
        std::vector<Detection> dets;
        // Every intersection trigger, including one with sign-provided depth,
        // requires a recent geometrically valid stop line from the point cloud.
        if (line_valid_ && fresh(line_stamp_)) {
            const bool has_sign = fresh(sign_input_stamp_);
            if (!has_sign || (sign_valid_ && std::abs(sign_input_stamp_ - line_stamp_) <= max_age_ &&
                              std::abs(sign_distance_ - line_distance_) <= sign_distance_tolerance_)) {
                try {
                    dets.push_back(has_sign ? line_detection(sign_distance_, sign_points_, sign_header_)
                                            : line_detection(line_distance_, line_points_, line_header_));
                    add_real_sign(dets);
                } catch (const std::exception& ex) {
                    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                                        "Stop-line input unavailable: %s", ex.what());
                }
            }
        }
        state_->on_intersection_detections(dets, fresh(sign_input_stamp_) && sign_valid_);
    }

    void on_line(const sensor_msgs::msg::PointCloud2& msg) {
        const double stamp = rclcpp::Time(msg.header.stamp).seconds();
        if (!fresh(stamp)) return;
        line_stamp_ = stamp;
        line_valid_ = false;
        try {
            sensor_msgs::msg::PointCloud2 cloud;
            if (msg.header.frame_id == base_) cloud = msg;
            else tf2::doTransform(msg, cloud, buffer_.lookupTransform(
                base_, msg.header.frame_id, rclcpp::Time(msg.header.stamp), rclcpp::Duration::from_seconds(0.05)));
            std::vector<StopLinePoint> points;
            if (!cloud.data.empty()) {
                sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z");
                for (; x != x.end(); ++x, ++y, ++z) {
                    if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z) &&
                        *x >= 0.0f && std::abs(*y) <= half_width_) points.push_back({*x, *y});
                }
            }
            const auto line = find_stop_line(points, min_points_, min_span_, line_tolerance_, max_slope_, max_gap_);
            if (line.valid) {
                line_valid_ = true;
                line_distance_ = line.distance;
                line_points_ = line.points;
                line_header_ = msg.header;
                line_header_.frame_id = base_;
            }
        } catch (const std::exception& ex) {
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                                "Stop-line input unavailable: %s", ex.what());
        }
        update_decision();
    }

    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<SharedState> state_;
    tf2_ros::Buffer buffer_;
    tf2_ros::TransformListener listener_;
    std::string base_, odom_, sign_class_;
    double max_age_, tf_tolerance_, sign_ttl_, min_score_, real_threshold_, half_width_;
    double min_span_, line_tolerance_, max_slope_, max_gap_, sign_distance_tolerance_;
    double line_stamp_ = -1e9, line_distance_ = 0, sign_distance_ = 0;
    int line_points_ = 0, sign_points_ = 0;
    bool line_valid_ = false, sign_valid_ = false;
    std_msgs::msg::Header line_header_, sign_header_;
    int min_points_;
    double real_sign_stamp_ = -1e9, sign_input_stamp_ = -1e9;
    rclcpp::Subscription<vision_interfaces::msg::DetectionArray>::SharedPtr signs_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lines_;
};
} // namespace bt
