// rl_controller.cpp
#include "nav2_rl_controller/rl_controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <cmath>

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
  name_ = name;
  node_ = parent.lock();
  logger_ = node_->get_logger();

  RCLCPP_INFO(logger_, "Configuring RLController: %s", name_.c_str());

  // read optional params
  node_->declare_parameter(name_ + ".rl_service_name", std::string("/rl_infer"));
  node_->declare_parameter(name_ + ".controller_timeout_ms", controller_timeout_ms_);
  node_->declare_parameter(name_ + ".lookahead_distance", lookahead_distance_);

  std::string rl_service_name;
  node_->get_parameter(name_ + ".rl_service_name", rl_service_name);
  node_->get_parameter(name_ + ".controller_timeout_ms", controller_timeout_ms_);
  node_->get_parameter(name_ + ".lookahead_distance", lookahead_distance_);

  rl_client_ = node_->create_client<nav2_rl_controller::srv::RLInfer>(rl_service_name);

  debug_pub_ = node_->create_publisher<geometry_msgs::msg::Twist>("rl_debug_cmd", 1);
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
}

double RLController::quaternion_to_yaw(const geometry_msgs::msg::Quaternion & q)
{
  double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

// Find a point on current_plan_ at about lookahead_distance ahead of robot_pose
bool RLController::find_lookahead_point(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  geometry_msgs::msg::PoseStamped & lookahead_pt,
  double lookahead_distance)
{
  if (current_plan_.poses.empty()) {
    return false;
  }
  // find closest index
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
  // walk forward from min_idx until distance >= lookahead_distance
  for (size_t i = min_idx; i < current_plan_.poses.size(); ++i) {
    double dx = current_plan_.poses[i].pose.position.x - robot_pose.pose.position.x;
    double dy = current_plan_.poses[i].pose.position.y - robot_pose.pose.position.y;
    double d = std::hypot(dx, dy);
    if (d >= lookahead_distance) {
      lookahead_pt = current_plan_.poses[i];
      return true;
    }
  }
  // fallback: last point
  lookahead_pt = current_plan_.poses.back();
  return true;
}

geometry_msgs::msg::TwistStamped RLController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.stamp = node_->now();

  // Build observation vector: [lookahead_x_rel, lookahead_y_rel, vel_lin, vel_ang]
  std::vector<double> obs;

  geometry_msgs::msg::PoseStamped lookahead_pt;
  bool have_lookahead = find_lookahead_point(pose, lookahead_pt, lookahead_distance_);

  double rel_x = 0.0, rel_y = 0.0;
  if (have_lookahead) {
    double dx = lookahead_pt.pose.position.x - pose.pose.position.x;
    double dy = lookahead_pt.pose.position.y - pose.pose.position.y;
    double yaw = quaternion_to_yaw(pose.pose.orientation);
    // rotate into robot frame: [x', y'] = R(-yaw) * [dx, dy]
    rel_x =  dx * std::cos(-yaw) - dy * std::sin(-yaw);
    rel_y =  dx * std::sin(-yaw) + dy * std::cos(-yaw);
  }

  double v_lin = velocity.linear.x;
  double v_ang = velocity.angular.z;

  obs.push_back(rel_x);
  obs.push_back(rel_y);
  obs.push_back(v_lin);
  obs.push_back(v_ang);

  // Build service request
  auto request = std::make_shared<nav2_rl_controller::srv::RLInfer::Request>();
  request->obs = obs;

  // Wait for service small timeout
  if (!rl_client_->wait_for_service(std::chrono::milliseconds(50))) {
    RCLCPP_WARN(logger_, "RL service unavailable (timeout). Falling back to zero cmd.");
    cmd.twist.linear.x = 0.0;
    cmd.twist.angular.z = 0.0;
    return cmd;
  }

  auto future = rl_client_->async_send_request(request);

  // Wait for response with our timeout
  auto status = future.wait_for(std::chrono::milliseconds((int)controller_timeout_ms_));
  if (status == std::future_status::timeout) {
    RCLCPP_WARN(logger_, "RL service call timed out. Falling back to zero cmd.");
    cmd.twist.linear.x = 0.0;
    cmd.twist.angular.z = 0.0;
    return cmd;
  }

  auto result = future.get();
  cmd.twist.linear.x = result->linear_x;
  cmd.twist.angular.z = result->angular_z;

  // publish debug
  geometry_msgs::msg::Twist dbg;
  dbg.linear.x = cmd.twist.linear.x;
  dbg.angular.z = cmd.twist.angular.z;
  debug_pub_->publish(dbg);

  return cmd;
}

void RLController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  (void)speed_limit; (void)percentage;
  // not used in this simple controller
}

} // namespace nav2_rl_controller

// Register plugin
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_rl_controller::RLController, nav2_core::Controller)
