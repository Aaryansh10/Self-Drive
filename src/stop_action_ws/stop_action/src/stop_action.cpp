#include <algorithm>
#include <cmath>
#include <chrono>
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
#include "fusion_interfaces/msg/tracked_obstacle_array.hpp"
#include "vision_interfaces/msg/detection_array.hpp"
#include "stop_action/stop_logic.hpp"

using det = fusion_interfaces::msg::TrackedObstacleArray;

namespace igvc_stop {
    using Stop = stop_interfaces::action::Stop;
    using GoalHandle = rclcpp_action::ServerGoalHandle<Stop>;

    enum class State {MONITORING, BRAKING, STOPPED};

    static const char * state_name(State s){
        switch(s) {
            case State::MONITORING : return "MONITORING";
            case State::BRAKING : return "BRAKING";
            case State::STOPPED : return "STOPPED";
        }
        return "UNKNOWN";
    }

    struct Track {
        std::deque<double> obstacle_s;
        std::string cls;
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
            det_sub = this->create_subscription<det>(detection_topic, qos, std::bind(&StopActionServer::detection_callback,this, std::placeholders::_1));
            odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(odom_topic, qos, std::bind(&StopActionServer::odom_callback,this, std::placeholders::_1));
            cmd_pub = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic, 10);

            action_server = rclcpp_action::create_server<Stop>(this, "stop", std::bind(&StopActionServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2), std::bind(&StopActionServer::handle_cancel, this, std::placeholders::_1), std::bind(&StopActionServer::handle_accepted, this, std::placeholders::_1));
            timer = rclcpp::create_timer(this, get_clock(), rclcpp::Duration::from_seconds(1.0 / control_rate_hz), std::bind(&StopActionServer::control_loop, this));

            RCLCPP_INFO(get_logger(), "Stop action server ready: %zu profile(s), a_comfort=%.2f a_max=%.2f t_lat=%.2f", profiles.size(), a_comfort, a_max, t_lat);
        }

        private:


        struct Candidate {
            std::string name;
            double d;
            double gap;
        };

        void declare_parameters() {
            declare_parameter<std::string>("detection_topic", "/obstacle_detections");
            declare_parameter<std::string>("odom_topic", "/odom");
            declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel_stop");
            declare_parameter<double>("control_rate_hz", 20.0);
            declare_parameter<double>("max_lateral_offset", 1.2);
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
           lm = get_parameter("max_lateral_offset").as_double();
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
            v_meas = msg->twist.twist.linear.x;
            w_meas = msg->twist.twist.angular.z;
        }

        void detection_callback(const det::SharedPtr m) {
            if (!have_odom) return;
            const auto n = this->now();
            for (const auto &o : m->obstacles) {
                if (profiles.find(o.class_name) == profiles.end()) continue;
                if (std::fabs(o.position_base.y) > lm) continue;   // in-lane check
                const double x = o.position_base.x;                // forward distance
                if (!std::isfinite(x) || x <= 0.0) continue;
                auto &t = tracks[static_cast<int>(o.id)];
                t.cls = o.class_name;
                if ((n - t.stamp).seconds() > det_timeout) t.obstacle_s.clear();
                t.obstacle_s.push_back(odometer + x - sensor_offset);
                while (static_cast<int>(t.obstacle_s.size()) > filter_window) t.obstacle_s.pop_front();
                t.filtered_s = median(std::vector<double>(t.obstacle_s.begin(), t.obstacle_s.end()));
                t.stamp = n;
            }
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
            last_distance = 0.0;
            tracks.clear();
            last_tick = this->now();
            last_feedback = this->now();
            ts = this->now();
            RCLCPP_INFO(get_logger(), "Goal accepted(hold_time=%.1f s, target_class='%s')", hold_time, target_class.c_str());
        }

        //Control Loop

        std::vector<Candidate> collect_candidates(double ma) const {
            std::vector<Candidate> out;
            const auto n = this->now();
            for (const auto &p : tracks) {
                const auto &t = p.second;
                if (!target_class.empty() && t.cls != target_class) continue;
                if ((n - t.stamp).seconds() > ma) continue;
                out.push_back({t.cls, t.filtered_s - odometer, target_gap(profiles.at(t.cls), target_fraction)});
            }
            return out;
        }

        void publish_cmd(double v_cmd) {
            geometry_msgs::msg::Twist cmd;
            cmd.linear.x = v_cmd;
            const double v = std::fabs(v_meas);
            if (v > 0.1) {
                cmd.angular.z = w_meas * std::min(1.0, v_cmd/v);
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
            const auto now = this->now();
            const double dt = std::clamp((now - last_tick).seconds(), 1e-3, 0.5);
            last_tick = now;

            if(goal_handle->is_canceling()) {
                finish(false, "cancelled", last_distance);
                return;
            }

            const double v = std::fabs(v_meas);
            Stop::Feedback fb;
            fb.class_name = brake_class;

            if(state == State::MONITORING) {
                const auto cands = collect_candidates(det_timeout);
                if (!cands.empty()) ts = now;
                else if ((now - ts).seconds() > 2.0) { 
                    finish(false, "obstacle lost", last_distance); return; }
                double worst_margin = std::numeric_limits<double>::infinity();
                const Candidate * trigger = nullptr;
                double trigger_dist = 0.0;
                for (const auto & c : cands) {
                    const double trig = trigger_distance(v, a_comfort, t_lat, c.gap);
                    const double margin = c.d - trig;
                    if (margin < worst_margin) {
                        worst_margin = margin;
                        trigger = &c;
                        trigger_dist = trig;
                    }
                    const double vmax = profiles.at(c.name).max_speed;
                    if (v > vmax) {
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Speed %.2f m/s exceeds profile '%s' max_speed %.2f: stop window not guaranteed",
            v, c.name.c_str(), vmax);
                    }
                }
                if(trigger) {
                    fb.class_name = trigger->name;
                    fb.distance = static_cast<float>(trigger->d);
                    fb.trigger_distance = static_cast<float>(trigger_dist);
                    last_distance = trigger->d;
                    if (worst_margin <= 0.0) {
                        state = State::BRAKING;
                        brake_class = trigger->name;
                        v_cmd = v;
                        trigger_speed = v;
                        trigger_odometer = odometer;
                        RCLCPP_INFO(get_logger(), "Brake trigger: %s at %.2f m (trigger %.2f m), speed %.2f m/s",
            brake_class.c_str(), trigger->d, trigger_dist, v);
                    }
                }
            }

            if(state == State::BRAKING) {
                const auto cands = collect_candidates(dr_timeout);
                double v_ref = std::numeric_limits<double>::infinity();
                const Candidate * nearest = nullptr;
                for(const auto & c : cands) {
                    const double r = reference_speed(c.d, v, a_comfort, t_lat, c.gap);
                    if (r < v_ref) {
                        v_ref = r;
                        nearest = &c;
                    }
                }
                if (!nearest) {
                    v_ref = v_cmd - a_comfort * dt;
                }
                else {
                    fb.class_name = nearest->name;
                    fb.distance = static_cast<float>(nearest->d);
                    last_distance = nearest->d;
                }
                v_cmd = next_command(v_cmd, v_ref, a_max, dt);
                publish_cmd(v_cmd);

                if(v < v_stop && v_cmd < v_stop) {
                    state = State::STOPPED;
                    stopped_at = now;
                    v_cmd = 0.0;
                    RCLCPP_INFO(get_logger(), "Stopped: gap %.2f m, stopping distance %.2f m",
          last_distance, odometer - trigger_odometer);
                }
            }

            else if(state == State::STOPPED) {
                v_cmd = 0.0;
                publish_cmd(0.0);
                fb.distance = static_cast<float>(last_distance);
                if((now - stopped_at).seconds() >= hold_time) {
                    finish(true, "stopped", last_distance);
                    return;
                }
            }

            if((now - last_feedback).seconds() >= 0.1) {
                last_feedback = now;
                auto msg = std::make_shared<Stop::Feedback>(fb);
                msg->state = state_name(state);
                msg->speed = static_cast<float>(v);
                msg->cmd_speed = static_cast<float>(v_cmd);
                goal_handle->publish_feedback(msg);
            }
        }

        //parameters
        std::string detection_topic, odom_topic, cmd_vel_topic;
        double control_rate_hz{20.0}, a_comfort{1.0}, lm{1.2}, a_max{2.5}, t_lat{0.15};
        double target_fraction{0.5}, v_stop{0.03}, sensor_offset{0.0};
        double det_timeout{0.5}, dr_timeout{3.0}, min_score{0.3};
        int filter_window{5};
        bool require_in_lane{true};
        std::unordered_map<std::string, Profile> profiles;

        //odometry
        bool have_odom{false};
        double last_x{0.0}, last_y{0.0}, odometer{0.0}, v_meas{0.0}, w_meas{0.0};

        //detections
        std::unordered_map<int, Track> tracks;

        //goal state
        std::shared_ptr<GoalHandle> goal_handle;
        State state{State::MONITORING};
        double hold_time{0.0}, v_cmd{0.0}, trigger_speed{0.0}, trigger_odometer{0.0}, last_distance{0.0};
        std::string target_class, brake_class;
        rclcpp::Time last_tick{0, 0, RCL_ROS_TIME}, last_feedback{0, 0, RCL_ROS_TIME}, stopped_at{0, 0, RCL_ROS_TIME}, ts{0, 0, RCL_ROS_TIME};

        //ROS handles
        rclcpp::Subscription<det>::SharedPtr det_sub;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub;
        rclcpp_action::Server<Stop>::SharedPtr action_server;
        rclcpp::TimerBase::SharedPtr timer;

};

}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<igvc_stop::StopActionServer>());
  rclcpp::shutdown();
  return 0;
}



            



    
