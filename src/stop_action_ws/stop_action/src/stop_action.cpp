#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
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
#include "stop_action/stop_logic.hpp"

namespace igvc_stop {

using Stop = stop_interfaces::action::Stop;
using GoalHandle = rclcpp_action::ServerGoalHandle<Stop>;

enum class State { MONITORING, BRAKING, STOPPED };

static const char * state_name(State s) {
    switch (s) {
        case State::MONITORING: return "MONITORING";
        case State::BRAKING: return "BRAKING";
        case State::STOPPED: return "STOPPED";
    }
    return "UNKNOWN";
}

class StopActionServer : public rclcpp::Node {
public:
    StopActionServer() : Node("stop_action_server") {
        declare_parameters();
        load_parameters();

        odom_sub = create_subscription<nav_msgs::msg::Odometry>(
            odom_topic, rclcpp::SensorDataQoS(),
            std::bind(&StopActionServer::odom_callback, this, std::placeholders::_1));
        cmd_pub = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic, 10);

        action_server = rclcpp_action::create_server<Stop>(
            this, "stop",
            std::bind(&StopActionServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&StopActionServer::handle_cancel, this, std::placeholders::_1),
            std::bind(&StopActionServer::handle_accepted, this, std::placeholders::_1));

        timer = rclcpp::create_timer(
            this, get_clock(), rclcpp::Duration::from_seconds(1.0 / control_rate_hz),
            std::bind(&StopActionServer::control_loop, this));

        RCLCPP_INFO(get_logger(),
            "Stop action server ready: %zu profile(s), a_comfort=%.2f a_max=%.2f t_lat=%.2f",
            profiles.size(), a_comfort, a_max, t_lat);
    }

private:
    // ------------------------------------------------------------------ parameters
    void declare_parameters() {
        declare_parameter<std::string>("odom_topic", "/odom");
        declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel_stop");
        declare_parameter<double>("control_rate_hz", 20.0);
        declare_parameter<double>("a_comfort", 1.0);
        declare_parameter<double>("a_max", 2.5);
        declare_parameter<double>("latency", 0.15);
        declare_parameter<double>("target_fraction", 0.5);
        declare_parameter<double>("stop_speed_threshold", 0.03);
        declare_parameter<double>("sensor_to_front_offset", 0.0);
        declare_parameter<double>("goal_timeout", 30.0);
        declare_parameter<std::vector<std::string>>("profile_names", std::vector<std::string>{});
        declare_parameter<double>("default_profile.min_distance", 0.5);
        declare_parameter<double>("default_profile.max_distance", 1.5);
        declare_parameter<double>("default_profile.max_speed", 2.2);
    }

    void load_parameters() {
        odom_topic = get_parameter("odom_topic").as_string();
        cmd_vel_topic = get_parameter("cmd_vel_topic").as_string();
        control_rate_hz = get_parameter("control_rate_hz").as_double();
        a_comfort = get_parameter("a_comfort").as_double();
        a_max = get_parameter("a_max").as_double();
        t_lat = get_parameter("latency").as_double();
        target_fraction = get_parameter("target_fraction").as_double();
        v_stop = get_parameter("stop_speed_threshold").as_double();
        sensor_offset = get_parameter("sensor_to_front_offset").as_double();
        goal_timeout = get_parameter("goal_timeout").as_double();

        if (control_rate_hz <= 0.0 || a_comfort <= 0.0 || a_max < a_comfort || t_lat < 0.0 || goal_timeout <= 0.0) {
            throw std::runtime_error("Invalid parameters: need control_rate_hz>0, a_comfort>0, a_max>=a_comfort, latency>=0, goal_timeout>0");
        }

        default_profile = read_profile("default_profile.", "default_profile");
        for (const auto & name : get_parameter("profile_names").as_string_array()) {
            if (name.empty()) continue;
            const std::string prefix = "profiles." + name + ".";
            declare_parameter<double>(prefix + "min_distance", default_profile.min_distance);
            declare_parameter<double>(prefix + "max_distance", default_profile.max_distance);
            declare_parameter<double>(prefix + "max_speed", default_profile.max_speed);
            profiles[name] = read_profile(prefix, name);
        }
    }

    Profile read_profile(const std::string & prefix, const std::string & label) {
        Profile p;
        p.min_distance = get_parameter(prefix + "min_distance").as_double();
        p.max_distance = get_parameter(prefix + "max_distance").as_double();
        p.max_speed = get_parameter(prefix + "max_speed").as_double();
        if (p.min_distance < 0.0 || p.max_distance < p.min_distance) {
            throw std::runtime_error("Incorrect distance parameters for profile: " + label);
        }
        return p;
    }

    // ------------------------------------------------------------------ odometry
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        const double x = msg->pose.pose.position.x;
        const double y = msg->pose.pose.position.y;
        if (have_odom) {
            odometer += std::hypot(x - last_x, y - last_y);
        }
        last_x = x;
        last_y = y;
        have_odom = true;
        v_meas = msg->twist.twist.linear.x;
        w_meas = msg->twist.twist.angular.z;
    }

    // ------------------------------------------------------------------ action callbacks
    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID &, std::shared_ptr<const Stop::Goal> goal) {
        if (goal_handle) {
            RCLCPP_WARN(get_logger(), "Rejecting goal: another goal is active");
            return rclcpp_action::GoalResponse::REJECT;
        }
        if (!(goal->hold_time >= 0.0F)) {  
            RCLCPP_WARN(get_logger(), "Rejecting goal: invalid hold_time");
            return rclcpp_action::GoalResponse::REJECT;
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle>) {
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandle> gh) {
        goal_handle = gh;
        const auto goal = gh->get_goal();
        hold_time = goal->hold_time;
        target_class = goal->target_class;

        const auto it = profiles.find(target_class);
        if (it != profiles.end()) {
            profile = it->second;
        } else {
            profile = default_profile;
            RCLCPP_WARN(get_logger(), "No profile for class '%s': using default_profile [%.2f, %.2f]",
                target_class.c_str(), profile.min_distance, profile.max_distance);
        }
        gap = target_gap(profile, target_fraction);

        const double d0 = static_cast<double>(goal->obstacle_distance);
        have_distance = std::isfinite(d0) && d0 > 0.0;
        if (have_distance) {
            obstacle_s = odometer + d0 - sensor_offset;
            last_distance = d0 - sensor_offset;
        } else {
            last_distance = 0.0;
            RCLCPP_WARN(get_logger(), "Goal has no obstacle_distance: braking to a halt immediately");
        }

        state = State::MONITORING;
        v_cmd = 0.0;
        trigger_speed = 0.0;
        trigger_odometer = 0.0;
        const auto n = this->now();
        goal_start = n;
        last_tick = n;
        last_feedback = n;
        RCLCPP_INFO(get_logger(), "Goal accepted (class='%s', distance=%.2f m, hold_time=%.1f s, target gap=%.2f m)",
            target_class.c_str(), have_distance ? last_distance : -1.0, hold_time, gap);
    }

    // ------------------------------------------------------------------ helpers
    void publish_cmd(double v_cmd_val) {
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = v_cmd_val;
        const double v = std::fabs(v_meas);
        if (v > 0.1) {
            cmd.angular.z = w_meas * std::min(1.0, v_cmd_val / v);
        }
        cmd_pub->publish(cmd);
    }

    void finish(bool success, const std::string & message) {
        auto result = std::make_shared<Stop::Result>();
        result->success = success;
        result->message = message;
        result->class_name = target_class;
        result->final_distance = static_cast<float>(have_distance ? last_distance : -1.0);
        result->stopping_distance = static_cast<float>(odometer - trigger_odometer);
        result->trigger_speed = static_cast<float>(trigger_speed);
        if (goal_handle->is_canceling()) {
            goal_handle->canceled(result);
        } else if (success) {
            goal_handle->succeed(result);
        } else {
            goal_handle->abort(result);
        }
        RCLCPP_INFO(get_logger(), "Goal finished: %s", message.c_str());
        goal_handle.reset();
    }

    void begin_braking(double v, double trig) {
        state = State::BRAKING;
        v_cmd = v;
        trigger_speed = v;
        trigger_odometer = odometer;
        RCLCPP_INFO(get_logger(), "Brake trigger: %s at %.2f m (trigger %.2f m), speed %.2f m/s",
            target_class.c_str(), have_distance ? last_distance : -1.0, trig, v);
    }

    // ------------------------------------------------------------------ control loop
    void control_loop() {
        if (!goal_handle) return;
        const auto now_t = this->now();

        if (goal_handle->is_canceling()) {
            finish(false, "cancelled");
            return;
        }
        if ((now_t - goal_start).seconds() > goal_timeout) {
            finish(false, "timeout: brake point never reached / never stopped");
            return;
        }
        if (!have_odom) return;

        const double dt = std::clamp((now_t - last_tick).seconds(), 1e-3, 0.5);
        last_tick = now_t;
        const double v = std::fabs(v_meas);

        if (have_distance) {
            last_distance = obstacle_s - odometer;
        }

        Stop::Feedback fb;
        fb.class_name = target_class;
        fb.distance = static_cast<float>(have_distance ? last_distance : -1.0);

        if (state == State::MONITORING) {
            if (!have_distance) {
                begin_braking(v, 0.0);
            } else {
                const double trig = trigger_distance(v, a_comfort, t_lat, gap);
                fb.trigger_distance = static_cast<float>(trig);
                if (v > profile.max_speed) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                        "Speed %.2f m/s exceeds max_speed %.2f of class '%s': stop window not guaranteed",
                        v, profile.max_speed, target_class.c_str());
                }
                if (last_distance <= trig) {
                    begin_braking(v, trig);
                }
            }
        }

        if (state == State::BRAKING) {
            const double v_ref = have_distance ?
                reference_speed(last_distance, v, a_comfort, t_lat, gap) :
                v_cmd - a_comfort * dt;  
            v_cmd = next_command(v_cmd, v_ref, a_max, dt);
            publish_cmd(v_cmd);

            if (v < v_stop && v_cmd < v_stop) {
                state = State::STOPPED;
                stopped_at = now_t;
                v_cmd = 0.0;
                RCLCPP_INFO(get_logger(), "Stopped: gap %.2f m, stopping distance %.2f m",
                    have_distance ? last_distance : -1.0, odometer - trigger_odometer);
            }
        } else if (state == State::STOPPED) {
            v_cmd = 0.0;
            publish_cmd(0.0);
            if ((now_t - stopped_at).seconds() >= hold_time) {
                finish(true, "stopped");
                return;
            }
        }

        if ((now_t - last_feedback).seconds() >= 0.1) {
            last_feedback = now_t;
            auto msg = std::make_shared<Stop::Feedback>(fb);
            msg->state = state_name(state);
            msg->speed = static_cast<float>(v);
            msg->cmd_speed = static_cast<float>(v_cmd);
            goal_handle->publish_feedback(msg);
        }
    }

    // ------------------------------------------------------------------ members
    // parameters
    std::string odom_topic, cmd_vel_topic;
    double control_rate_hz{20.0}, a_comfort{1.0}, a_max{2.5}, t_lat{0.15};
    double target_fraction{0.5}, v_stop{0.03}, sensor_offset{0.0}, goal_timeout{30.0};
    std::unordered_map<std::string, Profile> profiles;
    Profile default_profile;

    // odometry
    bool have_odom{false};
    double last_x{0.0}, last_y{0.0}, odometer{0.0}, v_meas{0.0}, w_meas{0.0};

    // current goal
    std::shared_ptr<GoalHandle> goal_handle;
    State state{State::MONITORING};
    Profile profile;
    std::string target_class;
    bool have_distance{false};
    double hold_time{0.0}, gap{0.0}, obstacle_s{0.0}, last_distance{0.0};
    double v_cmd{0.0}, trigger_speed{0.0}, trigger_odometer{0.0};
    rclcpp::Time goal_start{0, 0, RCL_ROS_TIME}, last_tick{0, 0, RCL_ROS_TIME},
        last_feedback{0, 0, RCL_ROS_TIME}, stopped_at{0, 0, RCL_ROS_TIME};

    // ROS handles
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub;
    rclcpp_action::Server<Stop>::SharedPtr action_server;
    rclcpp::TimerBase::SharedPtr timer;
};

}  

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<igvc_stop::StopActionServer>());
    rclcpp::shutdown();
    return 0;
}