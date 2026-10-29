#include <atomic>
#include <thread>
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "intersection_interfaces_cpp/action/intersection.hpp"
#include "intersection_cpp/intersection_planner.hpp"   // <- adjust to wherever your planner header lives
#include "rclcpp_action/rclcpp_action.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

namespace {

double yaw_from_quat(const geometry_msgs::msg::Quaternion & q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

tf2::TimePoint to_tf2_time(const rclcpp::Time & t) {
    if (t.nanoseconds() == 0) return tf2::TimePointZero;   // "latest available"
    return tf2::TimePoint(std::chrono::nanoseconds(t.nanoseconds()));
}

}  // namespace

class IntersectionServer : public rclcpp::Node {
    public:
        using Action = intersection_interfaces_cpp::action::Intersection;
        using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;

        IntersectionServer() : Node("intersection_server") {
            load_params();

            tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

            // transient_local: a controller that starts late still receives the latest path
            path_pub_ = create_publisher<nav_msgs::msg::Path>(
                path_topic_, rclcpp::QoS(1).reliable().transient_local()
            );

            using namespace std::placeholders;
            this->action_server_ = rclcpp_action::create_server<Action>(
                this, 
                "intersection", 
                std::bind(&IntersectionServer::handle_goal, this, _1, _2),
                std::bind(&IntersectionServer::handle_cancel, this, _1),
                std::bind(&IntersectionServer::handle_accepted, this, _1)
            );

            costmap_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
                costmap_topic_, 1,
                [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
                    std::lock_guard<std::mutex> lock(costmap_mutex_);
                    latest_costmap_ = msg;
                }
            );

            RCLCPP_INFO(get_logger(), "Intersection action server started (path -> '%s', frame '%s')",
                path_topic_.c_str(), global_frame_.c_str());
        }

        ~IntersectionServer() override {
            if (exec_thread_.joinable()) exec_thread_.join();
        }
    
    private:
        rclcpp_action::Server<Action>::SharedPtr action_server_;
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_; 
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;   // non-const: publish() mutates
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
        nav_msgs::msg::OccupancyGrid::SharedPtr latest_costmap_;
        std::mutex costmap_mutex_;
        std::string costmap_topic_;

        intersection::IntersectionPlanner planner_;
        std::thread exec_thread_;
        std::atomic<bool> busy_{false};

        // --- parameters ---
        std::string path_topic_;
        std::string base_frame_;
        std::string global_frame_;          // fixed frame the path is published/tracked in
        double control_rate_{20.0};
        double goal_tolerance_{0.5};        // [m] distance to the final pose that counts as success
        double max_path_deviation_{1.5};    // [m] abort if the robot strays further than this
        double timeout_{60.0};              // [s]
        double tf_timeout_{0.2};            // [s]
        bool abort_if_blocked_{false};      // painted stop/cross lines also count as "blocked"
        double blocked_abort_fraction_{0.2};

        void load_params() {
            path_topic_ = declare_parameter("path_topic", std::string("intersection_path"));
            base_frame_ = declare_parameter("base_frame", std::string("link_base"));
            global_frame_ = declare_parameter("global_frame", std::string("odom"));
            control_rate_ = declare_parameter("control_rate", 20.0);
            goal_tolerance_ = declare_parameter("goal_tolerance", 0.5);
            costmap_topic_ = declare_parameter("costmap_topic", std::string("/lane_map/white"));
            max_path_deviation_ = declare_parameter("max_path_deviation", 1.5);
            timeout_ = declare_parameter("timeout", 60.0);
            tf_timeout_ = declare_parameter("tf_timeout", 0.2);
            abort_if_blocked_ = declare_parameter("abort_if_blocked", false);
            blocked_abort_fraction_ = declare_parameter("blocked_abort_fraction", 0.2);

            intersection::Params p;   // defaults come from the planner header
            p.occupied_threshold = declare_parameter("planner.occupied_threshold", p.occupied_threshold);
            p.lane_center_ratio = declare_parameter("planner.lane_center_ratio", p.lane_center_ratio);
            p.width_mismatch_tolerance = declare_parameter("planner.width_mismatch_tolerance", p.width_mismatch_tolerance);
            p.min_line_pts = declare_parameter("planner.min_line_pts", p.min_line_pts);
            p.ransac_iters = declare_parameter("planner.ransac_iters", p.ransac_iters);
            p.max_heading_error = declare_parameter("planner.max_heading_error", p.max_heading_error);
            p.line_inlier_tol = declare_parameter("planner.line_inlier_tol", p.line_inlier_tol);
            p.roi_back = declare_parameter("planner.roi_back", p.roi_back);
            p.roi_front = declare_parameter("planner.roi_front", p.roi_front);
            p.fit_x_front = declare_parameter("planner.fit_x_front", p.fit_x_front);
            p.max_line_gap = declare_parameter("planner.max_line_gap", p.max_line_gap);
            p.fallback_distance_to_inters = declare_parameter("planner.fallback_distance_to_inters", p.fallback_distance_to_inters);
            p.min_turning_radius = declare_parameter("planner.min_turning_radius", p.min_turning_radius);
            p.exit_length = declare_parameter("planner.exit_length", p.exit_length);
            p.merge_length = declare_parameter("planner.merge_length", p.merge_length);
            p.path_resolution = declare_parameter("planner.path_resolution", p.path_resolution);
            p.robot_radius = declare_parameter("planner.robot_radius", p.robot_radius);
            planner_.setParams(p);
        }

        rclcpp_action::GoalResponse handle_goal(
            const rclcpp_action::GoalUUID& uuid, 
            std::shared_ptr<const Action::Goal> goal
        )
        {
            (void) uuid;
            if (goal->turn_direction > Action::Goal::TURN_RIGHT) {
                RCLCPP_WARN(get_logger(), "Rejected goal: Invalid turn direction %u", goal->turn_direction);
                return rclcpp_action::GoalResponse::REJECT;
            }
            if (!(goal->road_width > 0.0f)) {
                RCLCPP_WARN(get_logger(), "Rejected goal: road_width must be > 0");
                return rclcpp_action::GoalResponse::REJECT;
            }
            std::lock_guard<std::mutex> lock(costmap_mutex_);
            if (!latest_costmap_ || latest_costmap_->data.empty()) {
                RCLCPP_WARN(get_logger(), "Rejected goal: no costmap received yet on %s", costmap_topic_.c_str());
                return rclcpp_action::GoalResponse::REJECT;
            }
            bool expected = false;
            if (!busy_.compare_exchange_strong(expected, true)) {
                RCLCPP_WARN(get_logger(), "Rejected goal: a maneuver is already running");
                return rclcpp_action::GoalResponse::REJECT;
            }
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }

        rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle> goal_handle) {
            RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
            (void) goal_handle;
            return rclcpp_action::CancelResponse::ACCEPT;
        }

        void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle) {
            // Never block the executor: run the maneuver in its own thread.
            if (exec_thread_.joinable()) exec_thread_.join();   // previous one has already finished
            exec_thread_ = std::thread([this, goal_handle]() { execute(goal_handle); });
        }

        void execute(const std::shared_ptr<GoalHandle> goal_handle) {
            // busy_ is released on every exit path
            struct BusyGuard {
                std::atomic<bool> & flag;
                ~BusyGuard() { flag = false; }
            } busy_guard{busy_};

            RCLCPP_INFO(this->get_logger(), "Executing turn");
            rclcpp::Rate loop_rate(std::max(1.0, control_rate_));

            const auto goal = goal_handle->get_goal();
            auto feedback = std::make_shared<Action::Feedback>();
            auto result = std::make_shared<Action::Result>();

            std::string turn_name;
            if (goal->turn_direction == Action::Goal::TURN_LEFT) {
                feedback->current_state = "Initiating left turn";
                turn_name = "left turn";
            } else if (goal->turn_direction == Action::Goal::TURN_RIGHT) {
                feedback->current_state = "Initiating right turn";
                turn_name = "right turn";
            } else {
                feedback->current_state = "Proceeding straight";
                turn_name = "straight";
            }
            feedback->progress = 0.0f;
            goal_handle->publish_feedback(feedback);

            // An empty path always tells the controller "maneuver over" and clears the latched path.
            auto abort_with = [&](const std::string & msg) {
                RCLCPP_WARN(get_logger(), "Intersection aborted: %s", msg.c_str());
                publish_path({});
                result->success = false;
                result->message = msg;
                goal_handle->abort(result);
            };

            // ---------------------------------------------------------- plan
            nav_msgs::msg::OccupancyGrid map;
            {
                std::lock_guard<std::mutex> lock(costmap_mutex_);
                map = *latest_costmap_;
            }
            const rclcpp::Time stamp(map.header.stamp);

            intersection::Pose2D robot_in_map;
            if (!lookup_robot_pose(map.header.frame_id, stamp, robot_in_map)) {
                abort_with("TF lookup failed: " + map.header.frame_id + " <- " + base_frame_);
                return;
            }

            const intersection::PlanResult plan = planner_.plan(
                map, robot_in_map, static_cast<intersection::Turn>(goal->turn_direction), goal->road_width);
            if (!plan.success) {
                abort_with("Planning failed: " + plan.message);
                return;
            }
            RCLCPP_INFO(get_logger(),
                "Planned %zu poses | %s W=%.2f | s_end=%.2f visible=%d R=%.2f blocked=%.0f%% | %s",
                plan.path.size(), turn_name.c_str(), goal->road_width, plan.debug.dist_to_intersection,
                static_cast<int>(plan.debug.intersection_visible), plan.debug.turn_radius,
                100.0 * plan.debug.blocked_fraction, plan.message.c_str());

            if (abort_if_blocked_ && plan.debug.blocked_fraction > blocked_abort_fraction_) {
                abort_with("Planned path is blocked in the costmap");
                return;
            }

            std::vector<intersection::Pose2D> path;
            if (!to_global_frame(plan.path, map.header.frame_id, stamp, path) || path.size() < 2) {
                abort_with("Could not transform path to " + global_frame_);
                return;
            }

            // cumulative arc length -> progress estimate
            std::vector<double> cum(path.size(), 0.0);
            for (size_t i = 1; i < path.size(); ++i) {
                cum[i] = cum[i - 1] + std::hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y);
            }
            const double total = std::max(cum.back(), 1e-3);

            publish_path(path);

            // ---------------------------------------------------------- monitor
            const rclcpp::Time t_start = now();
            size_t idx = 0;

            while (rclcpp::ok()) {
                if (goal_handle->is_canceling()) {
                    publish_path({});
                    result->success = false;
                    result->message = "Canceled";
                    goal_handle->canceled(result);
                    return;
                }
                if ((now() - t_start).seconds() > timeout_) {
                    abort_with("Timeout while executing the maneuver");
                    return;
                }

                intersection::Pose2D rp;
                if (!lookup_robot_pose(global_frame_, rclcpp::Time(0, 0, RCL_ROS_TIME), rp)) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for TF %s <- %s",
                        global_frame_.c_str(), base_frame_.c_str());
                    loop_rate.sleep();
                    continue;
                }

                // closest path pose, searched forward only (progress is monotonic)
                double best = 1e9;
                size_t best_i = idx;
                const size_t last = std::min(path.size() - 1, idx + 60);
                for (size_t i = idx; i <= last; ++i) {
                    const double d = std::hypot(path[i].x - rp.x, path[i].y - rp.y);
                    if (d < best) { best = d; best_i = i; }
                }
                idx = best_i;

                if (best > max_path_deviation_) {
                    abort_with("Robot deviated " + std::to_string(best) + " m from the path");
                    return;
                }

                feedback->current_state = "Executing " + turn_name;
                feedback->progress = static_cast<float>(cum[idx] / total);
                goal_handle->publish_feedback(feedback);

                const double to_goal = std::hypot(path.back().x - rp.x, path.back().y - rp.y);
                if (to_goal < goal_tolerance_) {
                    feedback->current_state = "Intersection completed";
                    feedback->progress = 1.0f;
                    goal_handle->publish_feedback(feedback);
                    publish_path({});
                    result->success = true;
                    result->message = "Intersection completed";
                    goal_handle->succeed(result);
                    RCLCPP_INFO(get_logger(), "Intersection completed in %.1f s", (now() - t_start).seconds());
                    return;
                }
                loop_rate.sleep();
            }

            // node is shutting down
            result->success = false;
            result->message = "Shutdown";
            if (goal_handle->is_active()) goal_handle->abort(result);
        }

        // Pose of base_frame_ expressed in `frame`. Tries the requested stamp first, then "latest".
        bool lookup_robot_pose(const std::string & frame, const rclcpp::Time & stamp, intersection::Pose2D & out) {
            geometry_msgs::msg::TransformStamped tf;
            try {
                tf = tf_buffer_->lookupTransform(
                    frame, base_frame_, to_tf2_time(stamp), tf2::durationFromSec(tf_timeout_));
            } catch (const tf2::TransformException &) {
                try {
                    tf = tf_buffer_->lookupTransform(
                        frame, base_frame_, tf2::TimePointZero, tf2::durationFromSec(tf_timeout_));
                } catch (const tf2::TransformException & ex) {
                    RCLCPP_DEBUG(get_logger(), "TF: %s", ex.what());
                    return false;
                }
            }
            out.x = tf.transform.translation.x;
            out.y = tf.transform.translation.y;
            out.yaw = yaw_from_quat(tf.transform.rotation);
            return true;
        }

        // The costmap may be robot-centric (moves with the robot). Convert once, here, to a
        // fixed frame so the path stays put while the robot drives.
        bool to_global_frame(
            const std::vector<intersection::Pose2D> & in, const std::string & map_frame,
            const rclcpp::Time & stamp, std::vector<intersection::Pose2D> & out
        ) {
            out = in;
            if (map_frame == global_frame_) return true;

            geometry_msgs::msg::TransformStamped tf;
            try {
                tf = tf_buffer_->lookupTransform(
                    global_frame_, map_frame, to_tf2_time(stamp), tf2::durationFromSec(tf_timeout_));
            } catch (const tf2::TransformException &) {
                try {
                    tf = tf_buffer_->lookupTransform(
                        global_frame_, map_frame, tf2::TimePointZero, tf2::durationFromSec(tf_timeout_));
                } catch (const tf2::TransformException & ex) {
                    RCLCPP_ERROR(get_logger(), "TF %s <- %s failed: %s",
                        global_frame_.c_str(), map_frame.c_str(), ex.what());
                    return false;
                }
            }
            const double tx = tf.transform.translation.x, ty = tf.transform.translation.y;
            const double yaw = yaw_from_quat(tf.transform.rotation);
            const double c = std::cos(yaw), s = std::sin(yaw);
            for (auto & p : out) {
                const double x = p.x, y = p.y;
                p.x = tx + c * x - s * y;
                p.y = ty + s * x + c * y;
                p.yaw += yaw;
            }
            return true;
        }

        // An empty vector publishes an empty path == "maneuver finished / cancelled".
        void publish_path(const std::vector<intersection::Pose2D> & path) {
            nav_msgs::msg::Path msg;
            msg.header.frame_id = global_frame_;
            msg.header.stamp = now();
            msg.poses.reserve(path.size());
            for (const auto & p : path) {
                geometry_msgs::msg::PoseStamped ps;
                ps.header = msg.header;
                ps.pose.position.x = p.x;
                ps.pose.position.y = p.y;
                ps.pose.orientation.z = std::sin(0.5 * p.yaw);
                ps.pose.orientation.w = std::cos(0.5 * p.yaw);
                msg.poses.push_back(ps);
            }
            path_pub_->publish(msg);
        }
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IntersectionServer>());
    rclcpp::shutdown();
    return 0;
}