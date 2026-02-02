#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <mutex>
#include <optional>

class VLACartesianExecutor : public rclcpp::Node
{
public:
  explicit VLACartesianExecutor(const rclcpp::NodeOptions & options)
  : Node("vla_cartesian_executor", options)
  {
    this->declare_parameter<std::string>("arm", "left");
    this->declare_parameter<double>("max_delta", 0.05);

    arm_       = this->get_parameter("arm").as_string();
    max_delta_ = this->get_parameter("max_delta").as_double();

    sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "/vla/base_point",
      10,
      std::bind(&VLACartesianExecutor::vlaCallback, this, std::placeholders::_1));

    // No timer: control loop runs in main thread so getCurrentState() can block while
    // a separate executor thread keeps the planning scene / joint_states updated.
    RCLCPP_INFO(get_logger(), "VLA Cartesian Executor constructed");
  }

  // ------------------------------------------------------------
  // Call AFTER executor is spinning
  // ------------------------------------------------------------
  void initMoveIt()
  {
    const std::string group_name =
      (arm_ == "right") ? "arm_right" : "arm_left";

    RCLCPP_INFO(get_logger(), "Initializing MoveIt group: %s", group_name.c_str());

    move_group_ =
      std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), group_name);

    move_group_->setMaxVelocityScalingFactor(0.3);
    move_group_->setMaxAccelerationScalingFactor(0.3);
    move_group_->setPlanningTime(3.0);

    // 🔑 MoveIt 2–correct state sync (allow time for joint_states timestamp to match sim time)
    RCLCPP_INFO(get_logger(), "Waiting for initial robot state (joint_states with recent timestamp)...");

    bool have_state = false;
    const int max_attempts = 40;
    const double wait_per_attempt = 1.0;  // seconds to wait for a "recent" state each try

    for (int i = 0; i < max_attempts && rclcpp::ok(); ++i) {
      auto state = move_group_->getCurrentState(wait_per_attempt);
      if (state) {
        have_state = true;
        break;
      }
      if ((i + 1) % 10 == 0) {
        RCLCPP_WARN(get_logger(),
          "Still waiting for robot state (attempt %d/%d). Check /joint_states and clock sync.",
          i + 1, max_attempts);
      }
      rclcpp::sleep_for(std::chrono::milliseconds(200));
    }

    if (!have_state) {
      RCLCPP_FATAL(get_logger(),
        "❌ No robot state received. Check /joint_states and /clock.");
      throw std::runtime_error("No initial robot state");
    }

    RCLCPP_INFO(get_logger(), "✅ MoveIt initialized and state synced");
  }

  // ------------------------------------------------------------
  // Called from main thread while a separate executor spins this node.
  // getCurrentState() can then succeed because joint_states are processed in the spinner thread.
  // ------------------------------------------------------------
  void controlLoop()
  {
    controlLoopImpl();
  }

private:
  // ------------------------------------------------------------
  void vlaCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_target_ = *msg;
  }

  // ------------------------------------------------------------
  // Servo-style loop (called from main thread)
  // ------------------------------------------------------------
  void controlLoopImpl()
  {
    if (!move_group_) return;

    std::optional<geometry_msgs::msg::PointStamped> target;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!latest_target_) return;
      target = latest_target_;
      latest_target_.reset();
    }

    // State is kept fresh by the separate executor thread spinning /joint_states
    auto state = move_group_->getCurrentState(1.0);
    if (!state) {
      RCLCPP_WARN(get_logger(), "No fresh robot state, skipping servo tick");
      return;
    }
    move_group_->setStartState(*state);

    auto current_pose_stamped = move_group_->getCurrentPose();
    const auto & current_pose = current_pose_stamped.pose;
    const std::string ee_frame = current_pose_stamped.header.frame_id;

    // VLA delta in base frame (from Python)
    double dx = clamp(target->point.x);
    double dy = clamp(target->point.y);
    double dz = clamp(target->point.z);

    // Log current state and intended delta for feasibility/debug
    RCLCPP_INFO(get_logger(),
      "[EE current] frame=%s pos=(%.4f, %.4f, %.4f) quat=(%.3f, %.3f, %.3f, %.3f)",
      ee_frame.c_str(),
      current_pose.position.x, current_pose.position.y, current_pose.position.z,
      current_pose.orientation.x, current_pose.orientation.y,
      current_pose.orientation.z, current_pose.orientation.w);
    RCLCPP_INFO(get_logger(),
      "[VLA delta] clamped (m): dx=%.4f dy=%.4f dz=%.4f (raw: %.4f, %.4f, %.4f)",
      dx, dy, dz,
      target->point.x, target->point.y, target->point.z);

    const double min_fraction = 0.05;  // execute if at least 5% of path is valid
    moveit_msgs::msg::RobotTrajectory traj;
    double fraction = 0.0;
    double scale = 1.0;
    const double scale_step = 0.5;  // try full step, then 50%, 25%, 12.5%
    const double min_scale = 0.1;

    // If full delta gives 0% path (collision/limits), retry with smaller step
    for (scale = 1.0; scale >= min_scale; scale *= scale_step) {
      geometry_msgs::msg::Pose target_pose = current_pose;
      target_pose.position.x = current_pose.position.x + scale * dx;
      target_pose.position.y = current_pose.position.y + scale * dy;
      target_pose.position.z = current_pose.position.z + scale * dz;

      std::vector<geometry_msgs::msg::Pose> waypoints{target_pose};
      traj = moveit_msgs::msg::RobotTrajectory();
      fraction = move_group_->computeCartesianPath(
        waypoints,
        0.01,   // eef_step
        0.0,    // jump_threshold
        traj);

      if (fraction >= min_fraction) break;
    }

    if (fraction < min_fraction) {
      double last_scale = (scale < 1.0) ? (scale / scale_step) : scale;
      geometry_msgs::msg::Pose fallback_pose = current_pose;
      fallback_pose.position.x = current_pose.position.x + last_scale * dx;
      fallback_pose.position.y = current_pose.position.y + last_scale * dy;
      fallback_pose.position.z = current_pose.position.z + last_scale * dz;

      RCLCPP_WARN(get_logger(),
        "[EE target attempted] frame=%s pos=(%.4f, %.4f, %.4f) scale=%.2f",
        ee_frame.c_str(),
        fallback_pose.position.x, fallback_pose.position.y, fallback_pose.position.z,
        last_scale);
      RCLCPP_INFO(get_logger(),
        "Cartesian path failed (%.1f%%), trying pose-target fallback (planned motion)...",
        fraction * 100.0);

      move_group_->setPoseTarget(fallback_pose);
      moveit::planning_interface::MoveGroupInterface::Plan plan_fallback;
      moveit_msgs::msg::MoveItErrorCodes result = move_group_->plan(plan_fallback);
      if (result.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
        move_group_->execute(plan_fallback);
        RCLCPP_INFO(get_logger(), "Pose-target fallback executed");
      } else {
        RCLCPP_WARN(get_logger(),
          "Pose-target fallback also failed, skipping (check kinematics/limits/collision)");
      }
      return;
    }

    // Log target we are executing
    RCLCPP_INFO(
      get_logger(),
      "[EE target]  frame=%s pos=(%.4f, %.4f, %.4f) quat=(%.3f, %.3f, %.3f, %.3f) scale=%.2f",
      ee_frame.c_str(),
      current_pose.position.x + scale * dx,
      current_pose.position.y + scale * dy,
      current_pose.position.z + scale * dz,
      current_pose.orientation.x, current_pose.orientation.y,
      current_pose.orientation.z, current_pose.orientation.w,
      scale);
    if (fraction < 0.95) {
      RCLCPP_INFO(get_logger(),
        "Executing partial Cartesian path (%.1f%%)", fraction * 100.0);
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = traj;

    move_group_->execute(plan);
  }

  // ------------------------------------------------------------
  double clamp(double v) const
  {
    if (v >  max_delta_) return  max_delta_;
    if (v < -max_delta_) return -max_delta_;
    return v;
  }

  // ------------------------------------------------------------
  std::string arm_;
  double max_delta_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_;

  std::mutex mutex_;
  std::optional<geometry_msgs::msg::PointStamped> latest_target_;
};

// ------------------------------------------------------------
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  // 🔑 use_sim_time must be set at construction
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
    rclcpp::Parameter("use_sim_time", true)
  });

  auto node = std::make_shared<VLACartesianExecutor>(opts);

  // Separate executor: dedicated thread only spins the node so planning scene / joint_states
  // stay updated. Main thread runs the control loop and getCurrentState() so it can block
  // while the spinner keeps the robot state fresh (fixes "recent timestamp" failures).
  rclcpp::executors::SingleThreadedExecutor spinner_executor;
  spinner_executor.add_node(node);

  std::thread spinner_thread([&spinner_executor]() {
    spinner_executor.spin();
  });

  // Wait for sim time
  RCLCPP_INFO(node->get_logger(), "Waiting for /clock...");
  while (rclcpp::ok() && node->get_clock()->now().nanoseconds() == 0) {
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }
  RCLCPP_INFO(node->get_logger(), "✅ Sim time active");

  RCLCPP_INFO(node->get_logger(), "Waiting 2 s for joint_states / move_group sync...");
  rclcpp::sleep_for(std::chrono::milliseconds(2000));

  node->initMoveIt();

  // Main thread: run control loop (getCurrentState() here can block; spinner keeps state updated)
  const auto loop_period = std::chrono::milliseconds(200);
  while (rclcpp::ok()) {
    node->controlLoop();
    rclcpp::sleep_for(loop_period);
  }

  rclcpp::shutdown();
  if (spinner_thread.joinable()) {
    spinner_thread.join();
  }
  return 0;
}
