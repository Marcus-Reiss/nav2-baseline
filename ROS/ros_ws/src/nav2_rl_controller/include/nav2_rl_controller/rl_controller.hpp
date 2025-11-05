// rl_controller.hpp
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "nav2_core/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "nav2_rl_controller/srv/rl_infer.hpp"  // generated from RLInfer.srv

namespace nav2_rl_controller
{

class RLController : public nav2_core::Controller
{
public:
  using Ptr = std::shared_ptr<RLController>;
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

  // --- Added: scan and odom subscriptions ---
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::vector<float> last_scan_;
  std::vector<float> last_sectors_;
  float scan_angle_min_{-M_PI};
  float scan_angle_max_{M_PI};
  float last_min_obst_{10.0f};
  float last_front_min_{10.0f};
  float last_path_min_{10.0f};
  geometry_msgs::msg::Pose robot_pose_;

  // parameters
  int n_sectors_{36};
  double controller_timeout_ms_ = 150.0; // configurable via params
  double lookahead_distance_ = 0.8;      // meters

  // helpers
  double quaternion_to_yaw(const geometry_msgs::msg::Quaternion & q);
  bool find_lookahead_point(const geometry_msgs::msg::PoseStamped & robot_pose,
                            geometry_msgs::msg::PoseStamped & lookahead_pt,
                            double lookahead_distance);

  std::vector<float> compressScan(const std::vector<float> & scan, int n_sectors, float default_value = 10.0f);
  float minInWindow(const std::vector<float> & sectors, int center_idx, int half_w);
  std::pair<float,float> computeLookaheadRel(const nav_msgs::msg::Path & plan,
                                             const geometry_msgs::msg::Pose & robot_pose,
                                             float lookahead_distance);

  // callbacks
  void scan_cb(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg);
};

} // namespace nav2_rl_controller
