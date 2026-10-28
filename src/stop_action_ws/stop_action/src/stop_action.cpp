#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "stop_interfaces/action/stop.hpp"
#include "vision_interfaces/msg/detection_array.hpp"
#include "stop_action/stop_logic.hpp"


using det = vision_interfaces::msg::DetectionArray;

namespace stop {
    using Stop = stop_interfaces::action::Stop;
    using GoalHandle = rclcpp_action::ServerGoalHandle<Stop>;

    enum class State {MONITORING, BRAKING, STOP};

    static const char * state_name(State s){
        switch(s) {
            case State::MONITORING : return "MONITORING";
            case State::BRAKING : return "BRAKING";
            case State::STOP : return "STOP";
        }
        return "UNKNOWN";
    }

    struct Track {
        std::deque<double> obstacle_s;
        rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
        double filtered_s{0.0};
    };

    class StopActionServer : public rclcpp::Node 
    {
        public:
        StopActionServer():Node("stop_action_server") {
            declare_parameters();
            load_parameters();

            const auto qos = rclcpp::SensorDataQoS();
            det_sub = this->create_subscription<det>("detection_topic", qos, std::bind(&StopActionServer::detection_callback,this, std::placeholders::_1));
            odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("odom_topic", qos, std::bind(&StopActionServer::odom_callback,this, std::placeholders::_1));
            cmd_pub = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel_topic", 10);

            action_server = rclcpp_action::create_server<Stop>(this, "stop", std::bind(&StopActionServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2), std::bind(&StopActionServer::handle_cancel, this, std::placeholders::_1), std::bind(&StopActionServer::handle_accepted, this, std::placeholders::_1));
            timer = create_wall_timer(std::chrono::duration<double>(1.0/control_rate_hz), std::bind(&StopActionServer::timer_callback, this));

            RCLCPP_INFO(get_logger(), "Stop action server ready: %zu profile(s), a_comfort=%.2f a_max=%.2f t_lat=%.2f", profiles.size(), a_comfort, a_max, t_lat);
        }

        private:
        void declare_parameters() {
            declare_parameter<std::string>("detection_topic", "/obstacle_detections");
            declare_parameter<std::string>("odom_topic", "/odom");
            declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel_stop");
            declare_parameter<double>("control_rate_hz", 20.0);
            declare_parameter<double>("a_comfort", 1.0);
            declare_parameter<double>("a_max", 2.5);
            declare_parameter<double>("latency", 0.15);
            declare_parameter<double>("target_fraction", 0.5);
            declare_parameter<double>("stop_speed_threshold", 0.03);
            declare_parameter<double>("sensor_to_front_offset", 0.0);
            declare_parameter<double>("detection_timeout", 0.5);
            declare_parameter<double>("dead_reckon_timeout", 3.0);
            declare_parameter<int>("filter_window", 5);
            declare_parameter<double>("min_score", 0.3);
            declare_parameter<bool>("require_in_lane", true);
            declare_parameter<std::vector<std::string>>("profile_names", std::vector<std::string>{});
        }

        void load_parameters() {
           detection_topic = get_parameter("detection_topic").as_string();
           odom_topic = get_parameter("odom_topic").as_string();
           cmd_vel_topic = get_parameter("cmd_vel_topic").as_string();
           control_rate_hz = get_parameter("control_rate_hz").as_double();
           a_comfort = get_parameter("a_comfort").as_double();
           a_max = get_parameter("a_max").as_double();
           t_lat = get_parameter("latency").as_double();
           target_fraction = get_parameter("target_fraction").as_double();
           v_stop = get_parameter("stop_speed_threshold").as_double();
           sensor_offset = get_parameter("sensor_to_front_offset").as_double();
           det_timeout = get_parameter("detection_timeout").as_double();
           dr_timeout = get_parameter("dead_reckon_timeout").as_double();
           filter_window = std::max<int>(1, get_parameter("filter_window").as_int());
           min_score = get_parameter("min_score").as_double();
           require_in_lane = get_parameter("require_in_lane").as_bool();
        
        if (control_rate_hz <= 0.0 || a_comfort <= 0.0 || a_max < a_comfort || t_lat < 0.0) {
            throw std::runtime_error("invalid parameters");
        }

        for (const auto &name : get_parameter("profile_names").as_string_array()) {
            const std::string prefix = "profiles." + name + ".";
            declare_parameter<double>(prefix + "min_distance", 0.0);
            declare_parameter<double>(prefix + "max_distance", 0.0);
            declare_parameter<double>(prefix + "max_speed", 2.2);
            Profile p;
            p.min_distance = get_parameter(prefix + "min_distance").as_double();
            p.max_distance = get_parameter(prefix + "max_distance").as_double();
            p.max_speed = get_parameter(prefix + "max_speed").as_double();
            if (p.min_distance < 0.0 || p.max_distance < p.min_distance)
            throw std::runtime_error("Incorrect distance parameters");
            profiles[name] = p;
        }
        if (profiles.empty())
        throw std::runtime_error("No profiles configured");

        }

        //Subscription callbacks

        void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
            const double x = msg->pose.pose.position.x;
            const double y = msg->pose.pose.position.y;
            if (have_odom) {
                odometer += std::hypot(x - last_x, y - last_y);
            }
            last_x = x;
            last_y = y;
            have_odom = true;
            v_ = msg->twist.twist.linear.x;
            w_ = msg->twist.twist.angular.z;
        }

        void detection_callback(const det::SharedPtr msg) {
            if (have_odom == false)
            return;
            if (!msg->depth_valid || !std::isfinite(msg->distance) || msg->distance <= 0.0F) 
            return;
            if (require_in_lane && !msg->is_in_lane) 
            return;
            if (msg->score < min_score)
            return;
            if (profiles.find(msg->class_name) == profiles.end()) 
            return;

            const auto now=Time.now();
            auto &t = tracks[msg->class_name];
            if ((now - t.stamp).seconds() > det_timeout)
            t.obstacle_s.clear();
            t.obstacle_s.push_back(odometer + static_cast<double>(msg->distance) - sensor_offset);
            while (static_cast<int>(t_obstacle_s.size()) > filter_window) {
                t.obstacle_s.pop_front();
            }
            t.filtered_s = median(std::vector<double>(t.obstacle_s.begin(), t.obstacle_s.end()));
            t.stamp = now;
        }

        //Action callbacks

        rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID &, std::shared_ptr<const Stop::Goal> goal) {
            if(goal_handle) {
                RCLCPP_WARN(get_logger(), "Rejecting goal: Another goal is active");
                return rclcpp_action::GoalResponse::REJECT;
            }
            if(goal->hold_time < 0.0F) {
                RCLCPP_WARN(get_logger(), "Rejecting goal: hold_time < 0");
                return rclcpp_action::GoalResponse::REJECT;
            }
            if(!goal->target_class.empty() && profiles.find(goal->target_class) == profiles.end()) {
                RCLCPP_WARN(get_logger(), "Rejecting goal: Unknown target class");
                return rclcpp_action::GoalResponse::REJECT;
            }
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }

        rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle>) {
            return rclcpp_action::CancelResponse::ACCEPT;
        }
        }

        void handle_accepted(const std::shared_ptr<GoalHandle> gh) {
            goal_handle = gh;
            const auto goal = gh->get_goal();
            hold_time = goal->hold_time;
            target_class = goal->target_class;
            state = State::MONITORING;
            v_cmd = 0.0;
            trigger_speed = 0.0;
            trigger_odometer = 0.0;
            brake_class.clear();
            last_tick = this.now();
            last_feedback = this.now();
            RCLCPP_INFO(get_logger(), "Goal accepted(hold_time=%.1f s, target_class='%s')", hold_time, target_class.c_str());
        }

        //Control Loop

        struct Candidate {
            std::string name;
            double d;
            double gap;
        };

        std::vector<Candidate> collect_candidates(double max_age) const {
            std::vector<Candidate> out;
            const auto now = this->now();
            for (const auto & [name,track] : tracks) {
                if(!target_class.empty() && name != target_class) 
                continue;
                if((now - track.stamp).seconds() > max_age)
                continue;
                out.push_back({name, track.filtered_s - odometer, target_gap(profiles.at(name), target_fraction)});
            }
            return out;
        }

        void publish_cmd(double v_cmd) {
            geometry_msgs::msg::Twist cmd;
            cmd.linear.x = v_cmd;
            const double v = std::fabs(v_);
            if (v > 0.1) {
                cmd.angular.z = w_ * std::min(1.0, v_cmd/v);
            }
            cmd_pub->publish(cmd);
        }

        void finish(bool success, const std::string & message, double final_distance) {
            auto result = std::make_shared<Stop::Result>();
            result->success = success;
            result->message = message;
            result->class_name = brake_class;
            result->final_distance = static_cast<float>(final_distance);
            result->stopping_distance = static_cast<float>(odometer - trigger_odometer);
            result->trigger_speed = static_cast<float>(trigger_speed);
            if(goal_handle->is_canceling()) {
                goal_handle->canceled(result);
            }
            else if(success) {
                goal_handle->succeed(result);
            }
            else {
                goal_handle->abort(result);
            }
            RCLCPP_INFO(get_logger(), "Goal finished : %s", message.c_str());
            goal_handle.reset();
        }

        void control_loop() {
            if (!goal_handle || !have_odom)
            return ;
            const auto now = this->now;
            const double dt = std::clamp((now - last_tick).seconds, 1e-3, 0.5);
            last_tick = now;

            if(goal_handle->is_canceling) {
                finish(false, "cancelled", last_distance);
                return;
            }

            const double v = std::fabs(v);
            Stop::Feedback fb;
            fb.class_name = brake_class;

            


        }


    }
