// rl_controller.cpp
#include "nav2_rl_controller/rl_controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <cmath>
#include <limits>
#include <chrono>
#include <functional>
#include <tf2/utils.h>

namespace nav2_rl_controller
{

RLController::RLController()
: logger_(rclcpp::get_logger("nav2_rl_controller"))
{}

void RLController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  (void)tf; (void)costmap_ros;
  name_ = name;
  node_ = parent.lock();
  logger_ = node_->get_logger();

  RCLCPP_INFO(logger_, "Configuring RLController: %s", name_.c_str());

  // parameters
  node_->declare_parameter(name_ + ".rl_service_name", std::string("/rl_infer"));
  node_->declare_parameter(name_ + ".controller_timeout_ms", controller_timeout_ms_);
  node_->declare_parameter(name_ + ".lookahead_distance", lookahead_distance_);
  node_->declare_parameter(name_ + ".n_sectors", n_sectors_);

  std::string rl_service_name;
  node_->get_parameter(name_ + ".rl_service_name", rl_service_name);
  node_->get_parameter(name_ + ".controller_timeout_ms", controller_timeout_ms_);
  node_->get_parameter(name_ + ".lookahead_distance", lookahead_distance_);
  node_->get_parameter(name_ + ".n_sectors", n_sectors_);

  rl_client_ = node_->create_client<nav2_rl_controller::srv::RLInfer>(rl_service_name);
  debug_pub_ = node_->create_publisher<geometry_msgs::msg::Twist>("rl_debug_cmd", 1);

  // subs
  scan_sub_ = node_->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", rclcpp::SensorDataQoS(),
    std::bind(&RLController::scan_cb, this, std::placeholders::_1));

  odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
    "/odom", rclcpp::QoS(10),
    std::bind(&RLController::odom_cb, this, std::placeholders::_1));

  last_scan_.clear();
  last_sectors_.assign(n_sectors_, 10.0f);
}

void RLController::cleanup()
{
  RCLCPP_INFO(logger_, "Cleaning RLController");
  rl_client_.reset();
  debug_pub_.reset();
  node_.reset();
}

void RLController::activate()
{
  RCLCPP_INFO(logger_, "Activating RLController");
}

void RLController::deactivate()
{
  RCLCPP_INFO(logger_, "Deactivating RLController");
}

void RLController::setPlan(const nav_msgs::msg::Path & path)
{
  current_plan_ = path;
  if (node_) {
    RCLCPP_INFO(node_->get_logger(), "RLController::setPlan called — plan size=%zu", current_plan_.poses.size());
  } else {
    RCLCPP_INFO(logger_, "RLController::setPlan called — plan size=%zu", current_plan_.poses.size());
  }
}

void RLController::scan_cb(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  size_t L = msg->ranges.size();
  last_scan_.resize(L);
  for (size_t i = 0; i < L; ++i) {
    float v = msg->ranges[i];
    if (!std::isfinite(v)) v = 10.0f;
    last_scan_[i] = v;
  }
  scan_angle_min_ = msg->angle_min;
  scan_angle_max_ = msg->angle_max;
  last_sectors_ = compressScan(last_scan_, n_sectors_, 10.0f);
  last_min_obst_ = *std::min_element(last_scan_.begin(), last_scan_.end());

  // compute front_min
  double ang_min = scan_angle_min_;
  double ang_max = scan_angle_max_;
  int n = n_sectors_;
  int front_idx = static_cast<int>(std::round((0.0 - ang_min) / (ang_max - ang_min + 1e-9) * (n - 1)));
  front_idx = std::max(0, std::min(n - 1, front_idx));
  int half_w = std::max(1, static_cast<int>(n * 60 / 360.0));
  last_front_min_ = minInWindow(last_sectors_, front_idx, half_w);

  // path_min (goal direction)
  if (!current_plan_.poses.empty()) {
    auto goal = current_plan_.poses.back().pose.position;
    double dx = goal.x - robot_pose_.position.x;
    double dy = goal.y - robot_pose_.position.y;
    double yaw = tf2::getYaw(robot_pose_.orientation);
    double curr_angle = std::atan2(dy, dx) - yaw;
    while (curr_angle > M_PI) curr_angle -= 2*M_PI;
    while (curr_angle < -M_PI) curr_angle += 2*M_PI;
    double frac = (curr_angle - ang_min) / (ang_max - ang_min + 1e-9);
    int sector_idx = std::clamp(static_cast<int>(std::round(frac * (n - 1))), 0, n - 1);
    int w = std::max(1, static_cast<int>(n * 20 / 360.0));
    last_path_min_ = minInWindow(last_sectors_, sector_idx, w);
  }
}

void RLController::odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  robot_pose_.position = msg->pose.pose.position;
  robot_pose_.orientation = msg->pose.pose.orientation;
}

double RLController::quaternion_to_yaw(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

bool RLController::find_lookahead_point(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  geometry_msgs::msg::PoseStamped & lookahead_pt,
  double lookahead_distance)
{
  if (current_plan_.poses.empty())
    return false;

  double min_dist = std::numeric_limits<double>::infinity();
  size_t min_idx = 0;
  for (size_t i = 0; i < current_plan_.poses.size(); ++i) {
    double dx = current_plan_.poses[i].pose.position.x - robot_pose.pose.position.x;
    double dy = current_plan_.poses[i].pose.position.y - robot_pose.pose.position.y;
    double d = std::hypot(dx, dy);
    if (d < min_dist) {
      min_dist = d;
      min_idx = i;
    }
  }
  for (size_t i = min_idx; i < current_plan_.poses.size(); ++i) {
    double dx = current_plan_.poses[i].pose.position.x - robot_pose.pose.position.x;
    double dy = current_plan_.poses[i].pose.position.y - robot_pose.pose.position.y;
    if (std::hypot(dx, dy) >= lookahead_distance) {
      lookahead_pt = current_plan_.poses[i];
      return true;
    }
  }
  lookahead_pt = current_plan_.poses.back();
  return true;
}

std::vector<float> RLController::compressScan(const std::vector<float> & scan, int n_sectors, float default_value)
{
  std::vector<float> sectors(n_sectors, default_value);
  if (scan.empty())
    return sectors;
  size_t L = scan.size();
  size_t step = std::max<size_t>(1, L / n_sectors);
  for (int i = 0; i < n_sectors; ++i) {
    size_t start = i * step;
    size_t end = (i < n_sectors - 1) ? (start + step) : L;
    auto it = std::min_element(scan.begin() + start, scan.begin() + end);
    sectors[i] = (it != scan.end()) ? *it : default_value;
  }
  return sectors;
}

float RLController::minInWindow(const std::vector<float> & sectors, int center_idx, int half_w)
{
  int start = std::max(0, center_idx - half_w);
  int end = std::min((int)sectors.size(), center_idx + half_w + 1);
  if (start >= end)
    return 10.0f;
  auto it = std::min_element(sectors.begin() + start, sectors.begin() + end);
  return (it != sectors.end()) ? *it : 10.0f;
}

std::pair<float,float> RLController::computeLookaheadRel(
  const nav_msgs::msg::Path & plan,
  const geometry_msgs::msg::Pose & robot_pose,
  float lookahead_distance)
{
  if (plan.poses.empty())
    return {0.0f, 0.0f};
  double rx = robot_pose.position.x;
  double ry = robot_pose.position.y;
  double yaw = tf2::getYaw(robot_pose.orientation);

  double min_d = 1e12;
  size_t min_idx = 0;
  for (size_t i = 0; i < plan.poses.size(); ++i) {
    double dx = plan.poses[i].pose.position.x - rx;
    double dy = plan.poses[i].pose.position.y - ry;
    double d = std::hypot(dx, dy);
    if (d < min_d) {
      min_d = d;
      min_idx = i;
    }
  }

  geometry_msgs::msg::Pose look_pt = plan.poses.back().pose;
  for (size_t j = min_idx; j < plan.poses.size(); ++j) {
    double px = plan.poses[j].pose.position.x;
    double py = plan.poses[j].pose.position.y;
    double d = std::hypot(px - rx, py - ry);
    if (d >= lookahead_distance) {
      look_pt = plan.poses[j].pose;
      break;
    }
  }

  double dx = look_pt.position.x - rx;
  double dy = look_pt.position.y - ry;
  float x_rel = dx * std::cos(-yaw) - dy * std::sin(-yaw);
  float y_rel = dx * std::sin(-yaw) + dy * std::cos(-yaw);
  return {x_rel, y_rel};
}

geometry_msgs::msg::TwistStamped RLController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  (void)velocity; (void)goal_checker;
  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.stamp = node_->now();

  // === Build 44-D observation ===
  std::vector<float> sectors_norm = last_sectors_;
  if (sectors_norm.empty()) sectors_norm.assign(n_sectors_, 10.0f);
  for (auto &v : sectors_norm) {
    if (!std::isfinite(v)) v = 10.0f;
    v = std::clamp(v, 0.0f, 10.0f) / 10.0f;
  }

  // compute goal direction
  double gx = 0.0, gy = 0.0;
  if (!current_plan_.poses.empty()) {
    gx = current_plan_.poses.back().pose.position.x;
    gy = current_plan_.poses.back().pose.position.y;
  }
  double rx = robot_pose_.position.x, ry = robot_pose_.position.y;
  double dx = gx - rx, dy = gy - ry;
  double dist = std::hypot(dx, dy);
  double yaw = tf2::getYaw(robot_pose_.orientation);
  double curr_angle = (dist > 0) ? std::atan2(dy, dx) - yaw : 0.0;
  while (curr_angle > M_PI) curr_angle -= 2*M_PI;
  while (curr_angle < -M_PI) curr_angle += 2*M_PI;

  double angle_sin = std::sin(curr_angle);
  double angle_cos = std::cos(curr_angle);

  float min_norm = std::clamp(last_min_obst_, 0.0f, 10.0f) / 10.0f;
  float front_norm = std::clamp(last_front_min_, 0.0f, 10.0f) / 10.0f;
  float path_norm = std::clamp(last_path_min_, 0.0f, 10.0f) / 10.0f;

  auto look_rel = computeLookaheadRel(current_plan_, robot_pose_, lookahead_distance_);

  std::vector<float> obs;
  obs.reserve(n_sectors_ + 8);
  obs.insert(obs.end(), sectors_norm.begin(), sectors_norm.end());
  obs.push_back(dist);
  obs.push_back(angle_sin);
  obs.push_back(angle_cos);
  obs.push_back(min_norm);
  obs.push_back(front_norm);
  obs.push_back(path_norm);
  obs.push_back(look_rel.first);
  obs.push_back(look_rel.second);

  // === Call RLInfer ===
  if (!rl_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_WARN(logger_, "RL service unavailable (timeout).");
    cmd.twist.linear.x = 0.0;
    cmd.twist.angular.z = 0.0;
    return cmd;
  }

  auto req = std::make_shared<nav2_rl_controller::srv::RLInfer::Request>();
  req->obs.assign(obs.begin(), obs.end());  // conversão automática float -> double
  auto future = rl_client_->async_send_request(req);

  auto status = future.wait_for(std::chrono::milliseconds((int)controller_timeout_ms_));
  if (status == std::future_status::timeout) {
    RCLCPP_WARN(logger_, "RL service call timed out.");
    cmd.twist.linear.x = 0.0;
    cmd.twist.angular.z = 0.0;
    return cmd;
  }

  auto res = future.get();
  cmd.twist.linear.x = res->linear_x;
  cmd.twist.angular.z = res->angular_z;

  RCLCPP_INFO(node_->get_logger(),
    "[RLController] RLInfer result: lin=%.3f, ang=%.3f",
    res->linear_x, res->angular_z);

  debug_pub_->publish(cmd.twist);

  return cmd;
}

void RLController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  (void)speed_limit; (void)percentage;
}

} // namespace nav2_rl_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_rl_controller::RLController, nav2_core::Controller)
