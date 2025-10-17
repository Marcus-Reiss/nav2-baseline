// rl_controller.hpp
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nav2_core/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "nav2_rl_controller/srv/rl_infer.hpp"  // generated from RLInfer.srv

namespace nav2_rl_controller
{

class RLController : public nav2_core::Controller
{
public:
  using Ptr = std::shared_ptr<RLController>();
  RLController();
  ~RLController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  rclcpp::Logger logger_;
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string name_;
  nav_msgs::msg::Path current_plan_;

  rclcpp::Client<nav2_rl_controller::srv::RLInfer>::SharedPtr rl_client_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr debug_pub_;

  // helper
  double quaternion_to_yaw(const geometry_msgs::msg::Quaternion & q);
  bool find_lookahead_point(const geometry_msgs::msg::PoseStamped & robot_pose,
                            geometry_msgs::msg::PoseStamped & lookahead_pt,
                            double lookahead_distance);
  double controller_timeout_ms_ = 150.0; // configurable via params later
  double lookahead_distance_ = 0.8; // meters (configurable)
};

} // namespace nav2_rl_controller
