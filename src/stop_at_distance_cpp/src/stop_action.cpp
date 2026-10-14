#include "stop_at_distance/stop_action_server.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace stop_at_distance
{

namespace
{

using ActionResult = StopActionServer::StopAction::Result;

static_assert(static_cast<uint8_t>(Outcome::SUCCESS) == ActionResult::OUTCOME_SUCCESS, "outcome code mismatch");
static_assert(static_cast<uint8_t>(Outcome::CANCELED) == ActionResult::OUTCOME_CANCELED, "outcome code mismatch");
static_assert(static_cast<uint8_t>(Outcome::TIMEOUT) == ActionResult::OUTCOME_TIMEOUT, "outcome code mismatch");
static_assert(static_cast<uint8_t>(Outcome::OVERSHOOT) == ActionResult::OUTCOME_OVERSHOOT, "outcome code mismatch");
static_assert(static_cast<uint8_t>(Outcome::SETTLE_TIMEOUT) == ActionResult::OUTCOME_SETTLE_TIMEOUT, "outcome code mismatch");
static_assert(static_cast<uint8_t>(Outcome::INTERNAL_FAULT) == ActionResult::OUTCOME_INTERNAL_FAULT, "outcome code mismatch");

std::string fixed(double value, const char * unit)
{
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%.3f %s", value, unit);
  return buf;
}

}
const char * to_string(Phase phase)
{
  switch (phase) {
    case Phase::IDLE:   return "IDLE";
    case Phase::BRAKE:  return "BRAKE";
    case Phase::SETTLE: return "SETTLE";
    case Phase::HOLD:   return "HOLD";
    case Phase::ABORT:  return "ABORT";
    default:            return "UNKNOWN";
  }
}

const char * to_string(Mode mode)
{
  switch (mode) {
    case Mode::IDLE:            return "IDLE";
    case Mode::BRAKE:           return "BRAKE";
    case Mode::HOLD:            return "HOLD";
    case Mode::CONTROLLED_STOP: return "CONTROLLED_STOP";
    default:                    return "UNKNOWN";
  }
}

const char * to_string(Outcome outcome)
{
  switch (outcome) {
    case Outcome::SUCCESS:        return "SUCCESS";
    case Outcome::CANCELED:       return "CANCELED";
    case Outcome::TIMEOUT:        return "TIMEOUT";
    case Outcome::OVERSHOOT:      return "OVERSHOOT";
    case Outcome::SETTLE_TIMEOUT: return "SETTLE_TIMEOUT";
    case Outcome::INTERNAL_FAULT: return "INTERNAL_FAULT";
    default:                      return "UNKNOWN";
  }
}

Mode phase_mode(Phase phase)
{
  switch (phase) {
    case Phase::BRAKE:  return Mode::BRAKE;
    case Phase::SETTLE:
    case Phase::HOLD:   return Mode::HOLD;
    case Phase::ABORT:  return Mode::CONTROLLED_STOP;
    case Phase::IDLE:
    default:            return Mode::IDLE;
  }
}

void SuccessChecker::configure(const Config & config)
{
  eval_config = config;
  reset();
}

void SuccessChecker::reset()
{
  start_time = -1.0;
  settle_start_time = -1.0;
}

SuccessChecker::Decision SuccessChecker::step(double time, double distance, double velocity)
{
  if (start_time < 0.0) {
    start_time = time;
  }

  if (std::fabs(velocity) < eval_config.stop_speed_epsilon) {
    if (settle_start_time < 0.0) {
      settle_start_time = time;
    }
  } else {
    settle_start_time = -1.0;
  }

  if (settle_start_time >= 0.0 && (time - settle_start_time) >= eval_config.settle_threshold) {
    if (distance < eval_config.lower_boundary) {
      return Decision::OVERSHOOT;
    }
    if (distance <= eval_config.upper_boundary) {
      return Decision::COMPLETED;
    }
  }

  if ((time - start_time) > eval_config.max_settle_time) {
    return Decision::SETTLE_TIMEOUT;
  }
  return Decision::PENDING;
}

StopActionServer::StopActionServer(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("stop_action_server_node", options)
{
  declare_parameters();
}

void StopActionServer::declare_parameters()
{
  declare_parameter<double>("control_rate_hz", params.control_rate_hz);
  declare_parameter<double>("feedback_rate_hz", params.feedback_rate_hz);
  declare_parameter<double>("mission_timeout", params.mission_timeout);
  declare_parameter<double>("band_min", params.band_min);
  declare_parameter<double>("band_max", params.band_max);
  declare_parameter<double>("stop_speed_epsilon", params.stop_speed_epsilon);
  declare_parameter<double>("settle_threshold", params.settle_threshold_s);
  declare_parameter<double>("max_settle_time", params.max_settle_time);
  declare_parameter<double>("hold_duration", params.hold_duration);
  declare_parameter<double>("abort_stop_timeout", params.abort_stop_timeout);
  declare_parameter<double>("approach_gain", params.approach_gain);
  declare_parameter<double>("max_speed", params.max_speed_mps);
  declare_parameter<std::string>("distance_topic", params.distance_topic);
  declare_parameter<std::string>("odom_topic", params.odom_topic);
  declare_parameter<std::string>("cmd_vel_topic", params.cmd_vel_topic);
}

bool StopActionServer::load_parameters()
{
  params.control_rate_hz = get_parameter("control_rate_hz").as_double();
  params.feedback_rate_hz = get_parameter("feedback_rate_hz").as_double();
  params.mission_timeout = get_parameter("mission_timeout").as_double();
  params.band_min = get_parameter("band_min").as_double();
  params.band_max = get_parameter("band_max").as_double();
  params.stop_speed_epsilon = get_parameter("stop_speed_epsilon").as_double();
  params.settle_threshold_s = get_parameter("settle_threshold").as_double();
  params.max_settle_time = get_parameter("max_settle_time").as_double();
  params.hold_duration = get_parameter("hold_duration").as_double();
  params.abort_stop_timeout = get_parameter("abort_stop_timeout").as_double();
  params.approach_gain = get_parameter("approach_gain").as_double();
  params.max_speed_mps = get_parameter("max_speed").as_double();
  params.distance_topic = get_parameter("distance_topic").as_string();
  params.odom_topic = get_parameter("odom_topic").as_string();
  params.cmd_vel_topic = get_parameter("cmd_vel_topic").as_string();

  const auto fail = [this](const char * why) {
      RCLCPP_ERROR(get_logger(), "Invalid parameters: %s", why);
      return false;
    };

  if (params.control_rate_hz < 1.0) {return fail("control_rate_hz must be >= 1");}
  if (params.feedback_rate_hz <= 0.0) {return fail("feedback_rate_hz must be > 0");}
  if (params.mission_timeout <= 0.0) {return fail("mission_timeout must be > 0");}
  if (!(params.band_min >= 0.0 && params.band_min < params.band_max)) {
    return fail("need 0 <= band_min < band_max");
  }
  if (params.stop_speed_epsilon <= 0.0) {return fail("stop_speed_epsilon must be > 0");}
  if (params.settle_threshold_s <= 0.0) {return fail("settle_threshold_s must be > 0");}
  if (params.max_settle_time <= params.settle_threshold_s) {
    return fail("max_settle_time must exceed settle_threshold_s");
  }
  if (params.hold_duration < 0.0) {return fail("hold_duration_s must be >= 0");}
  if (params.abort_stop_timeout <= 0.0) {return fail("abort_stop_timeout_s must be > 0");}
  if (params.max_speed_mps <= 0.0 || params.approach_gain <= 0.0) {
    return fail("max_speed and approach_gain must be > 0");
  }
  return true;
}

StopActionServer::CallbackReturn StopActionServer::on_configure(const rclcpp_lifecycle::State &)
{
  if (!load_parameters()) {
    return CallbackReturn::FAILURE;
  }

  action_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  state_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  timer_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions state_opts;
  state_opts.callback_group = state_group_;

  distance_sub_ = create_subscription<std_msgs::msg::Float32>(
    params.distance_topic, 10,
    [this](std_msgs::msg::Float32::ConstSharedPtr msg) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      current_distance_m_ = static_cast<double>(msg->data);
      have_distance_ = true;
    },
    state_opts);

  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    params.odom_topic, 10,
    [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      current_speed_mps_ = msg->twist.twist.linear.x;
      have_speed_ = true;
    },
    state_opts);

  cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(params.cmd_vel_topic, 10);

  action_server_ = rclcpp_action::create_server<StopAction>(
    this, "stop_at_distance",
    std::bind(&StopActionServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
    std::bind(&StopActionServer::handle_cancel, this, std::placeholders::_1),
    std::bind(&StopActionServer::handle_accepted, this, std::placeholders::_1),
    rcl_action_server_get_default_options(), action_group_);

  RCLCPP_INFO(get_logger(), "Configured (band %.2f-%.2f m, %.0f Hz).",
    params.band_min, params.band_max, params.control_rate_hz);
  return CallbackReturn::SUCCESS;
}

StopActionServer::CallbackReturn StopActionServer::on_activate(const rclcpp_lifecycle::State & state)
{
  rclcpp_lifecycle::LifecycleNode::on_activate(state);

  control_timer_ = rclcpp::create_timer(
    this, get_clock(), rclcpp::Duration::from_seconds(1.0 / params.control_rate_hz),
    std::bind(&StopActionServer::control_tick, this), timer_group_);

  active_ = true;
  RCLCPP_INFO(get_logger(), "Activated.");
  return CallbackReturn::SUCCESS;
}

StopActionServer::CallbackReturn StopActionServer::on_deactivate(const rclcpp_lifecycle::State & state)
{
  active_ = false;                       
  if (control_timer_) {
    control_timer_->cancel();
  }
  release_mission("server deactivated");  
  control_timer_.reset();

  rclcpp_lifecycle::LifecycleNode::on_deactivate(state);
  RCLCPP_INFO(get_logger(), "Deactivated.");
  return CallbackReturn::SUCCESS;
}

StopActionServer::CallbackReturn StopActionServer::on_cleanup(const rclcpp_lifecycle::State &)
{
  action_server_.reset();
  distance_sub_.reset();
  odom_sub_.reset();
  cmd_vel_pub_.reset();
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_distance_m_ = std::numeric_limits<double>::quiet_NaN();
    current_speed_mps_ = std::numeric_limits<double>::quiet_NaN();
    have_distance_ = false;
    have_speed_ = false;
  }
  return CallbackReturn::SUCCESS;
}

StopActionServer::CallbackReturn StopActionServer::on_shutdown(const rclcpp_lifecycle::State &)
{
  active_ = false;
  if (control_timer_) {
    control_timer_->cancel();
  }
  release_mission("node shutting down");
  control_timer_.reset();
  return CallbackReturn::SUCCESS;
}

void StopActionServer::release_mission(const std::string & reason)
{
  std::lock_guard<std::mutex> lock(mission_mutex_);
  goal_reserved_ = false;
  if (active_mission_.goal_handle) {
    if (cmd_vel_pub_ && cmd_vel_pub_->is_activated()) {
      cmd_vel_pub_->publish(geometry_msgs::msg::Twist());  
    }
    finish_mission(Outcome::INTERNAL_FAULT, reason);
  }
}

rclcpp_action::GoalResponse StopActionServer::handle_goal(
  const rclcpp_action::GoalUUID &, std::shared_ptr<const StopAction::Goal>)
{
  if (!active_) {
    RCLCPP_WARN(get_logger(), "Goal rejected: server is not active.");
    return rclcpp_action::GoalResponse::REJECT;
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!have_distance_ || !have_speed_) {
      RCLCPP_WARN(get_logger(), "Goal rejected: no distance/odometry received from the sim yet.");
      return rclcpp_action::GoalResponse::REJECT;
    }
  }

  std::lock_guard<std::mutex> lock(mission_mutex_);
  if (active_mission_.goal_handle || goal_reserved_) {
    RCLCPP_WARN(get_logger(), "Goal rejected: another stop mission is active.");
    return rclcpp_action::GoalResponse::REJECT;
  }
  goal_reserved_ = true;
  RCLCPP_INFO(get_logger(), "Goal accepted.");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse StopActionServer::handle_cancel(
  const std::shared_ptr<GoalHandle> goal_handle)
{
  std::lock_guard<std::mutex> lock(mission_mutex_);
  if (active_mission_.goal_handle &&
    active_mission_.goal_handle->get_goal_id() == goal_handle->get_goal_id())
  {
    RCLCPP_INFO(get_logger(), "Cancel requested.");
    return rclcpp_action::CancelResponse::ACCEPT;   
  }
  return rclcpp_action::CancelResponse::REJECT;
}

void StopActionServer::handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
{
  const double now_s = now().seconds();
  std::lock_guard<std::mutex> lock(mission_mutex_);
  goal_reserved_ = false;

  if (!active_) {
    auto result = std::make_shared<StopAction::Result>();
    result->success = false;
    result->outcome_code = static_cast<uint8_t>(Outcome::INTERNAL_FAULT);
    result->final_distance = std::numeric_limits<float>::quiet_NaN();
    result->message = "server deactivated before the goal started";
    goal_handle->abort(result);
    return;
  }

  active_mission_ = ActiveMission();
  active_mission_.goal_handle = goal_handle;
  active_mission_.goal_id = rclcpp_action::to_string(goal_handle->get_goal_id());
  active_mission_.start_time = now_s;
  active_mission_.phase_entry_time = now_s;
  active_mission_.last_feedback_time = now_s;
  active_mission_.target_band_min = params.band_min;
  active_mission_.target_band_max = params.band_max;
  active_mission_.timeout = params.mission_timeout;

  SuccessChecker::Config cfg;
  cfg.lower_boundary = active_mission_.target_band_min;
  cfg.upper_boundary = active_mission_.target_band_max;
  cfg.stop_speed_epsilon = params.stop_speed_epsilon;
  cfg.settle_threshold = params.settle_threshold_s;
  cfg.max_settle_time = params.max_settle_time;
  success_checker_.configure(cfg);

  transition_phase(Phase::BRAKE, "goal accepted");
}

void StopActionServer::control_tick()
{
  const double now_s = now().seconds();

  double dist = std::numeric_limits<double>::quiet_NaN();
  double vel = std::numeric_limits<double>::quiet_NaN();
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    dist = current_distance_m_;
    vel = current_speed_mps_;
  }

  std::lock_guard<std::mutex> lock(mission_mutex_);

  if (!active_mission_.goal_handle) {
    return;
  }
  if (!active_mission_.goal_handle->is_active()) {
    active_mission_ = ActiveMission();   
    return;
  }

  active_mission_.last_distance_m = dist;

  if (active_mission_.current_phase != Phase::ABORT) {
    const double elapsed = now_s - active_mission_.start_time;
    const Phase phase = active_mission_.current_phase;

    if (active_mission_.goal_handle->is_canceling()) {
      begin_stop(Outcome::CANCELED, "cancel requested");
    } else if (elapsed > active_mission_.timeout) {
      begin_stop(Outcome::TIMEOUT, "mission timeout after " + fixed(elapsed, "s"));
    } else if ((phase == Phase::BRAKE || phase == Phase::SETTLE) &&
      dist < active_mission_.target_band_min)
    {
      begin_stop(Outcome::OVERSHOOT, "distance " + fixed(dist, "m") + " is below hard floor " +
        fixed(active_mission_.target_band_min, "m"));
    }
  }

  geometry_msgs::msg::Twist cmd;

  switch (active_mission_.current_phase) {
    case Phase::BRAKE:
    {
      const double target = 0.5 * (active_mission_.target_band_min + active_mission_.target_band_max);
      if (std::fabs(vel) < params.stop_speed_epsilon && dist <= active_mission_.target_band_max) {
        transition_phase(Phase::SETTLE, "stopped inside band");
      } else {
        cmd.linear.x = std::clamp(params.approach_gain * (dist - target), 0.0, params.max_speed_mps);
      }
      break;
    }

    case Phase::SETTLE:
    {
      switch (success_checker_.step(now_s, dist, vel)) {
        case SuccessChecker::Decision::COMPLETED:
          transition_phase(Phase::HOLD, "stop confirmed at " + fixed(dist, "m"));
          break;
        case SuccessChecker::Decision::OVERSHOOT:
          begin_stop(Outcome::OVERSHOOT, "settled at " + fixed(dist, "m") + ", inside the band floor");
          break;
        case SuccessChecker::Decision::SETTLE_TIMEOUT:
          begin_stop(Outcome::SETTLE_TIMEOUT, "did not settle inside the band in time (last " +
            fixed(dist, "m") + ")");
          break;
        case SuccessChecker::Decision::PENDING:
          break;
      }
      break;
    }

    case Phase::HOLD:
    {
      if ((now_s - active_mission_.phase_entry_time) >= params.hold_duration) {
        finish_mission(Outcome::SUCCESS, "stopped at " + fixed(active_mission_.last_distance_m, "m"));
      }
      break;
    }

    case Phase::ABORT:
    {
      const double in_abort = now_s - active_mission_.phase_entry_time;
      const bool stopped = std::fabs(vel) < params.stop_speed_epsilon;
      if (stopped) {
        finish_mission(active_mission_.pending_outcome, active_mission_.pending_reason);
      } else if (in_abort > params.abort_stop_timeout) {
        finish_mission(active_mission_.pending_outcome,
          active_mission_.pending_reason + " (standstill not confirmed)");
      }
      break;
    }

    case Phase::IDLE:
    default:
      break;
  }

  if (cmd_vel_pub_ && cmd_vel_pub_->is_activated()) {
    cmd_vel_pub_->publish(cmd);
  }

  publish_feedback(now_s, dist, vel);
}

void StopActionServer::transition_phase(Phase new_phase, const std::string & reason)
{
  const Phase old_phase = active_mission_.current_phase;
  if (old_phase == new_phase) {
    return;
  }
  const double t = now().seconds();
  active_mission_.current_phase = new_phase;
  active_mission_.phase_entry_time = t;

  if (new_phase == Phase::BRAKE) {
    active_mission_.brake_start_time = t;
  } else if (new_phase == Phase::SETTLE) {
    success_checker_.reset();   
    if (!active_mission_.time_to_stop_recorded) {
      active_mission_.time_to_stop = t - active_mission_.brake_start_time;
      active_mission_.time_to_stop_recorded = true;
    }
  }

  RCLCPP_INFO(get_logger(), "[%s] %s -> %s (mode %s): %s",
    active_mission_.goal_id.c_str(), to_string(old_phase), to_string(new_phase),
    to_string(phase_mode(new_phase)), reason.c_str());
}

void StopActionServer::begin_stop(Outcome outcome, const std::string & reason)
{
  if (active_mission_.current_phase == Phase::ABORT) {
    return;   
  }
  active_mission_.pending_outcome = outcome;
  active_mission_.pending_reason = reason;
  RCLCPP_WARN(get_logger(), "[%s] stopping mission: %s (%s)",
    active_mission_.goal_id.c_str(), to_string(outcome), reason.c_str());
  transition_phase(Phase::ABORT, reason);
}

void StopActionServer::finish_mission(Outcome outcome, const std::string & reason)
{
  const auto handle = active_mission_.goal_handle;
  if (!handle) {
    return;
  }

  auto result = std::make_shared<StopAction::Result>();
  result->success = (outcome == Outcome::SUCCESS);
  result->outcome_code = static_cast<uint8_t>(outcome);
  result->final_distance = static_cast<float>(active_mission_.last_distance_m);
  result->time_to_stop = static_cast<float>(active_mission_.time_to_stop);
  result->message = reason;

  if (handle->is_active()) {
    if (outcome == Outcome::SUCCESS) {
      handle->succeed(result);
    } else if (outcome == Outcome::CANCELED && handle->is_canceling()) {
      handle->canceled(result);
    } else {
      handle->abort(result);
    }
  }

  RCLCPP_INFO(get_logger(), "[%s] mission finished: %s (%s)",
    active_mission_.goal_id.c_str(), to_string(outcome), reason.c_str());

  active_mission_ = ActiveMission();   
}

void StopActionServer::publish_feedback(double now_s, double dist, double vel)
{
  if (!active_mission_.goal_handle) {
    return;   
  }
  if ((now_s - active_mission_.last_feedback_time) < (1.0 / params.feedback_rate_hz)) {
    return;
  }
  active_mission_.last_feedback_time = now_s;

  const double target = 0.5 * (active_mission_.target_band_min + active_mission_.target_band_max);
  auto feedback = std::make_shared<StopAction::Feedback>();
  feedback->current_distance = static_cast<float>(dist);
  feedback->current_velocity = static_cast<float>(vel);
  feedback->distance_to_target = static_cast<float>(dist - target);
  feedback->phase = static_cast<uint8_t>(active_mission_.current_phase);
  active_mission_.goal_handle->publish_feedback(feedback);
}

}  
RCLCPP_COMPONENTS_REGISTER_NODE(stop_at_distance::StopActionServer)