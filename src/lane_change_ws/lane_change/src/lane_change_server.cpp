// rclcpp action server wrapping the pure-C++ lane_change core.
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "lane_change/core.hpp"
#include "lane_change_interfaces/action/lane_change.hpp"
#ifdef HAVE_NAV2_MSGS
#include "nav2_msgs/msg/speed_limit.hpp"
#endif
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/int32.hpp"

using Action = lane_change_interfaces::action::LaneChange;
using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;
namespace lc = lane_change;

static const char* phase_name(lc::Phase p) {
  switch (p) {
    case lc::Phase::IDLE: return "IDLE";
    case lc::Phase::WAITING: return "WAITING";
    case lc::Phase::CHANGING: return "CHANGING";
    case lc::Phase::SETTLING: return "SETTLING";
    case lc::Phase::DONE: return "DONE";
    default: return "FAILED";
  }
}

class LaneChangeServer : public rclcpp::Node {
 public:
  LaneChangeServer() : Node("lane_change_server") {
    // ---- core parameters
#define PD(f) params_.f = declare_parameter<double>(#f, params_.f)
#define PB(f) params_.f = declare_parameter<bool>(#f, params_.f)
    PD(lane_width); PD(robot_width); PD(front_offset); PD(body_clear_margin); PB(distance_is_from_base);
    PD(completion_distance); PD(window_min); PD(window_max); PB(completion_at_center);
    PD(nominal_length); PD(tracking_lag_s); PD(line_smooth); PD(max_lat_accel); PB(allow_late_compress);
    PD(path_length); PD(min_turn_radius); PD(speed_ratio); PD(min_speed); PD(max_speed); PD(max_accel);
    PD(lookahead_min); PD(lookahead_gain); PD(lookahead_max); PD(max_wz);
    PD(settle_lat_tol); PD(settle_yaw_tol); PD(settle_distance); PD(settle_timeout);
    PD(min_lateral_shift); PD(max_shift_factor); PD(lost_timeout); PD(wait_timeout); PD(max_change_time);
#undef PD
#undef PB
    // ---- node parameters
    base_frame_ = declare_parameter<std::string>("base_frame", "link_base");
    // "path": publish a nav_msgs/Path in the odom frame for Nav2 to follow (no cmd_vel). "cmd_vel": drive directly.
    slow_down_ = declare_parameter<bool>("slow_down", true);  // publish a Nav2 speed limit during the lane change
    auto speed_topic = declare_parameter<std::string>("speed_limit_topic", "/speed_limit");
    output_path_ = declare_parameter<std::string>("output_mode", "path") == "path";
    path_frame_ = declare_parameter<std::string>("path_frame", "");  // "" = frame_id of the odometry message
    rate_hz_ = declare_parameter<double>("rate_hz", 20.0);
    odom_timeout_ = declare_parameter<double>("odom_timeout", 0.5);
    use_stamped_ = declare_parameter<bool>("use_stamped_cmd", false);
    auto odom_topic = declare_parameter<std::string>("odom_topic", "/odom");
    auto c1 = declare_parameter<std::string>("centerline_1_topic", "/centerline/lane_1");
    auto c2 = declare_parameter<std::string>("centerline_2_topic", "/centerline/lane_2");
    auto lane_topic = declare_parameter<std::string>("current_lane_topic", "/current_lane");
    auto cmd_topic = declare_parameter<std::string>("cmd_vel_topic", "/lane_change/cmd_vel");
    auto active_topic = declare_parameter<std::string>("active_topic", "/lane_change/active");
    auto path_topic = declare_parameter<std::string>("path_topic", "/lane_change/path");

    auto sensor_qos = rclcpp::SensorDataQoS();  // best-effort: compatible with reliable AND best-effort publishers
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, sensor_qos, [this](nav_msgs::msg::Odometry::SharedPtr m) { on_odom(*m); });
    line_sub_[0] = create_subscription<nav_msgs::msg::Path>(
        c1, sensor_qos, [this](nav_msgs::msg::Path::SharedPtr m) { on_line(0, *m); });
    line_sub_[1] = create_subscription<nav_msgs::msg::Path>(
        c2, sensor_qos, [this](nav_msgs::msg::Path::SharedPtr m) { on_line(1, *m); });
    lane_sub_ = create_subscription<std_msgs::msg::Int32>(
        lane_topic, 10, [this](std_msgs::msg::Int32::SharedPtr m) { std::lock_guard<std::mutex> l(mtx_); cur_lane_ = m->data; });

    if (use_stamped_) cmd_pub_s_ = create_publisher<geometry_msgs::msg::TwistStamped>(cmd_topic, 10);
    else cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_topic, 10);
    active_pub_ = create_publisher<std_msgs::msg::Bool>(active_topic, 10);
#ifdef HAVE_NAV2_MSGS
    speed_pub_ = create_publisher<nav2_msgs::msg::SpeedLimit>(speed_topic, 1);
#else
    (void)speed_topic;
    if (slow_down_) RCLCPP_WARN(get_logger(), "built without nav2_msgs: no /speed_limit will be published");
#endif
    path_pub_ = create_publisher<nav_msgs::msg::Path>(path_topic, 10);

    server_ = rclcpp_action::create_server<Action>(
        this, "lane_change",
        [this](const rclcpp_action::GoalUUID&, std::shared_ptr<const Action::Goal>) {
          if (busy_.exchange(true)) {
            RCLCPP_WARN(get_logger(), "goal rejected: a lane change is already running");
            return rclcpp_action::GoalResponse::REJECT;
          }
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](std::shared_ptr<GoalHandle>) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](std::shared_ptr<GoalHandle> gh) {
          std::thread([this, gh]() {
            execute(gh);
            busy_ = false;
          }).detach();
        });
    RCLCPP_INFO(get_logger(), "lane_change action server ready (action name: lane_change)");
  }

 private:
  void on_odom(const nav_msgs::msg::Odometry& m) {
    const auto& q = m.pose.pose.orientation;
    double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    std::lock_guard<std::mutex> l(mtx_);
    odom_.x = m.pose.pose.position.x;
    odom_.y = m.pose.pose.position.y;
    odom_.yaw = yaw;
    odom_.speed = std::hypot(m.twist.twist.linear.x, m.twist.twist.linear.y);
    odom_frame_ = m.header.frame_id;
    odom_time_ = now();
    have_odom_ = true;
  }
  void on_line(int idx, const nav_msgs::msg::Path& m) {
    if (!m.header.frame_id.empty() && m.header.frame_id != base_frame_)
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "centreline %d frame is '%s' but '%s' is expected (no TF transform is applied)", idx + 1,
                           m.header.frame_id.c_str(), base_frame_.c_str());
    lc::Line l;
    l.reserve(m.poses.size());
    for (const auto& p : m.poses) l.push_back(lc::Pt{p.pose.position.x, p.pose.position.y});
    std::lock_guard<std::mutex> g(mtx_);
    lines_[idx] = std::move(l);
    seq_[idx]++;
  }

  // Nav2 SpeedLimit: absolute m/s while the lane change runs; 0.0 = no limit (restores the normal speed)
  void set_speed_limit(bool on, double v = 0.0) {
#ifdef HAVE_NAV2_MSGS
    if (!slow_down_ || on == limit_on_) return;
    nav2_msgs::msg::SpeedLimit m;
    m.header.stamp = now();
    m.percentage = false;
    m.speed_limit = on ? v : 0.0;
    speed_pub_->publish(m);
    limit_on_ = on;
#else
    (void)on;
    (void)v;
#endif
  }
  void publish_active(bool on) {
    std_msgs::msg::Bool b;
    b.data = on;
    active_pub_->publish(b);
  }
  void publish_cmd(double v, double wz) {
    if (use_stamped_) {
      geometry_msgs::msg::TwistStamped t;
      t.header.stamp = now();
      t.header.frame_id = base_frame_;
      t.twist.linear.x = v;
      t.twist.angular.z = wz;
      cmd_pub_s_->publish(t);
    } else {
      geometry_msgs::msg::Twist t;
      t.linear.x = v;
      t.angular.z = wz;
      cmd_pub_->publish(t);
    }
  }
  // path in the odom/global frame (what Nav2 needs). Poses get a heading along the path.
  void publish_path_global(const lc::Line& l, const lc::OdomIn& od, const std::string& frame) {
    nav_msgs::msg::Path p;
    p.header.stamp = now();
    p.header.frame_id = frame;
    const double c = std::cos(od.yaw), s = std::sin(od.yaw);
    std::vector<lc::Pt> g;
    for (const auto& pt : l) g.push_back(lc::Pt{od.x + c * pt.x - s * pt.y, od.y + s * pt.x + c * pt.y});
    double yaw = od.yaw;
    for (size_t i = 0; i < g.size(); ++i) {
      if (i + 1 < g.size()) yaw = std::atan2(g[i + 1].y - g[i].y, g[i + 1].x - g[i].x);
      geometry_msgs::msg::PoseStamped ps;
      ps.header = p.header;
      ps.pose.position.x = g[i].x;
      ps.pose.position.y = g[i].y;
      ps.pose.orientation.z = std::sin(yaw / 2);
      ps.pose.orientation.w = std::cos(yaw / 2);
      p.poses.push_back(ps);
    }
    path_pub_->publish(p);
  }
  void publish_path(const lc::Line& l) {
    nav_msgs::msg::Path p;
    p.header.stamp = now();
    p.header.frame_id = base_frame_;
    for (const auto& pt : l) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = p.header;
      ps.pose.position.x = pt.x;
      ps.pose.position.y = pt.y;
      ps.pose.orientation.w = 1.0;
      p.poses.push_back(ps);
    }
    path_pub_->publish(p);
  }

  void finish(const std::shared_ptr<GoalHandle>& gh, lc::Code code, const std::string& msg,
              const lc::Maneuver* M) {
    auto res = std::make_shared<Action::Result>();
    res->code = static_cast<uint8_t>(code);
    res->message = msg;
    res->body_clear_distance = M ? M->body_clear_distance() : -1.0;
    res->in_window = M ? M->in_window() : false;
    publish_active(false);
    set_speed_limit(false);
    if (code == lc::Code::SUCCESS) gh->succeed(res);
    else if (code == lc::Code::CANCELED) gh->canceled(res);
    else gh->abort(res);
    RCLCPP_INFO(get_logger(), "lane change finished: code=%d '%s' clear_dist=%.2f in_window=%d",
                static_cast<int>(code), msg.c_str(), res->body_clear_distance, res->in_window);
  }

  static double lateral_dist(const lc::Line& l) {
    if (l.size() < 2) return 1e9;
    lc::Polyline pl;
    if (!pl.build(l, 3.0, 0.0, 0.4)) return 1e9;
    return std::fabs(pl.project(lc::Pt{0.0, 0.0}).signed_dist);
  }
  static int infer_lane(const lc::Line& a, const lc::Line& b) {
    double da = lateral_dist(a), db = lateral_dist(b);
    if (da > 1e8 || db > 1e8) return 0;
    return da <= db ? 1 : 2;
  }

  void execute(std::shared_ptr<GoalHandle> gh) {
    const auto goal = gh->get_goal();
    lc::GoalIn g;
    g.obstacle_distance = goal->obstacle_distance;
    g.completion_distance = goal->completion_distance;
    g.target_speed = goal->target_speed;
    g.target_lane_free = goal->target_lane_free;

    lc::OdomIn od;
    int lane = 0;
    lc::Line first_line, second_line;
    {
      std::lock_guard<std::mutex> l(mtx_);
      if (!have_odom_ || (now() - odom_time_).seconds() > odom_timeout_) {
        finish(gh, lc::Code::NO_DATA, "no recent odometry", nullptr);
        return;
      }
      od = odom_;
      lane = cur_lane_;
      first_line = lines_[0];
      second_line = lines_[1];
    }
    if (lane != 1 && lane != 2) {  // no current-lane message: the lane whose centreline is closer to the robot is the current one
      lane = infer_lane(first_line, second_line);
      if (lane == 0) {
        finish(gh, lc::Code::NO_DATA, "no current-lane message and the two centrelines are not both available", nullptr);
        return;
      }
      RCLCPP_WARN(get_logger(), "no current-lane message received: inferred lane %d from the centrelines", lane);
    }
    const int tgt = (lane == 1) ? 1 : 0;  // index of the OTHER lane's centreline
    RCLCPP_INFO(get_logger(), "lane change goal: lane %d -> %d, obstacle %.2f m", lane, tgt + 1,
                g.obstacle_distance);

    lc::Maneuver M(params_);
    std::string msg;
    lc::Code c = M.start(g, od, now().seconds(), &msg);
    if (c != lc::Code::SUCCESS) {
      finish(gh, c, msg, nullptr);
      return;
    }

    uint64_t last_seq;
    {
      std::lock_guard<std::mutex> l(mtx_);
      last_seq = seq_[tgt];  // only lines that arrive AFTER the goal count as fresh
    }
    rclcpp::Rate rate(rate_hz_);
    while (rclcpp::ok()) {
      if (gh->is_canceling()) {
        M.cancel();
        finish(gh, lc::Code::CANCELED, "canceled", &M);
        return;
      }
      lc::Line fresh;
      bool has_fresh = false;
      bool odom_ok = true;
      std::string frame;
      {
        std::lock_guard<std::mutex> l(mtx_);
        od = odom_;
        frame = odom_frame_;
        odom_ok = (now() - odom_time_).seconds() <= odom_timeout_;
        if (seq_[tgt] != last_seq) {
          fresh = lines_[tgt];
          last_seq = seq_[tgt];
          has_fresh = true;
        }
      }
      if (!odom_ok) {
        finish(gh, lc::Code::NO_DATA, "odometry lost", &M);
        return;
      }
      lc::StepOut o = M.step(od, has_fresh ? &fresh : nullptr, now().seconds());
      if (o.command_active) {
        if (output_path_) {
          publish_path_global(o.path, od, path_frame_.empty() ? frame : path_frame_);
        } else {
          publish_cmd(o.v, o.wz);
          publish_path(o.path);
        }
      }
      if (o.command_active) set_speed_limit(true, M.target_speed());
      publish_active(o.command_active);

      auto fb = std::make_shared<Action::Feedback>();
      fb->phase = phase_name(o.phase);
      fb->remaining_to_obstacle = o.remaining_to_obstacle;
      fb->lateral_error = o.lateral_error;
      fb->progress = o.progress;
      gh->publish_feedback(fb);

      if (o.finished) {
        finish(gh, o.code, o.message, &M);
        return;
      }
      rate.sleep();
    }
    finish(gh, lc::Code::CANCELED, "shutdown", &M);
  }

  lc::Params params_;
  std::string base_frame_;
  double rate_hz_ = 20.0, odom_timeout_ = 0.5;
  bool use_stamped_ = false, output_path_ = true, slow_down_ = true, limit_on_ = false;
  std::string path_frame_, odom_frame_ = "odom";
  std::mutex mtx_;
  lc::OdomIn odom_;
  rclcpp::Time odom_time_{0, 0, RCL_ROS_TIME};
  bool have_odom_ = false;
  int cur_lane_ = 0;
  std::array<lc::Line, 2> lines_;
  std::array<uint64_t, 2> seq_{{0, 0}};
  std::atomic<bool> busy_{false};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::array<rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr, 2> line_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr lane_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_pub_s_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr active_pub_;
#ifdef HAVE_NAV2_MSGS
  rclcpp::Publisher<nav2_msgs::msg::SpeedLimit>::SharedPtr speed_pub_;
#endif
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp_action::Server<Action>::SharedPtr server_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LaneChangeServer>();
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
