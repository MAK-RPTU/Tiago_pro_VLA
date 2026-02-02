import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from play_motion2_msgs.action import PlayMotion2
from rclpy.action import ActionClient
from fastmcp import FastMCP
import threading
import time
import math

# Initialize FastMCP specifically for the Holland Robot
mcp = FastMCP("HollandRobotController")

class HollandMcpNode(Node):
    def __init__(self):
        super().__init__('holland_mcp_bridge')
        
        # Publisher for movement commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for AMCL pose (accurate localization in map frame)
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, 
            '/amcl_pose', 
            self.amcl_callback, 
            10
        )
        
        # Action client for navigation
        self.nav_action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        
        # Action client for play_motion2 (arm movements)
        self.play_motion_client = ActionClient(self, PlayMotion2, '/play_motion2')
        
        self.current_pos = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self.latest_amcl_msg = None
        self.amcl_msg_received = False
        self.get_logger().info("Holland Robot Native Bridge Initialized")
        self.get_logger().info("Subscribed to /amcl_pose for accurate localization")
        self.get_logger().info("Navigation action client initialized")
        self.get_logger().info("PlayMotion2 action client initialized")

    def amcl_callback(self, msg):
        """Update the robot's current position from AMCL pose (more accurate)."""
        self.latest_amcl_msg = msg
        self.amcl_msg_received = True
        self._update_pos_from_amcl(msg)
    
    def _update_pos_from_amcl(self, msg):
        """Helper method to update position from AMCL message."""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        
        # Convert quaternion to yaw (rotation around z-axis)
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        w = orient.w
        x = orient.x
        y = orient.y
        z = orient.z
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        self.current_pos["x"] = round(pos.x, 3)
        self.current_pos["y"] = round(pos.y, 3)
        self.current_pos["yaw"] = round(yaw, 3)
        
        # Print the pose for debugging
        self.get_logger().info(
            f"AMCL Pose - X: {self.current_pos['x']:.3f}, "
            f"Y: {self.current_pos['y']:.3f}, "
            f"Yaw: {self.current_pos['yaw']:.3f} rad ({math.degrees(yaw):.1f} deg)"
        )
    
    def get_latest_amcl_pose(self):
        """Process the latest AMCL message to update position."""
        # Process the latest message we've received
        if self.latest_amcl_msg is not None:
            self._update_pos_from_amcl(self.latest_amcl_msg)

    def execute_move(self, linear_x, angular_z, duration):
        """Execute a timed movement and then force a stop."""
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        
        # Start moving
        self.cmd_pub.publish(msg)
        time.sleep(duration)
        
        # Force Stop (Safety)
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (roll, pitch, yaw) to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw
    
    def navigate_to_pose(self, x, y, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timeout=60.0):
        """Navigate to a specified pose using Nav2."""
        # Wait for action server to be available
        if not self.nav_action_client.wait_for_server(timeout_sec=5.0):
            return False, "Navigation action server not available"
        
        # Convert Euler angles to quaternion
        qx, qy, qz, qw = self.euler_to_quaternion(roll, pitch, yaw)
        
        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = float(z)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw
        
        # Send goal
        self.get_logger().info(f"Sending navigation goal to X={x}, Y={y}, Z={z}, Yaw={yaw}")
        send_goal_future = self.nav_action_client.send_goal_async(goal_msg)
        
        # Wait for goal to be accepted (polling since we're in a thread)
        start_time = time.time()
        while not send_goal_future.done() and (time.time() - start_time) < 5.0:
            time.sleep(0.1)
        
        if not send_goal_future.done():
            return False, "Failed to send navigation goal (timeout)"
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            return False, "Navigation goal was rejected"
        
        # Wait for result
        self.get_logger().info("Navigation goal accepted, waiting for completion...")
        get_result_future = goal_handle.get_result_async()
        
        start_time = time.time()
        while not get_result_future.done() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not get_result_future.done():
            # Cancel the goal if timeout
            cancel_future = goal_handle.cancel_goal_async()
            time.sleep(0.5)  # Give it time to cancel
            return False, f"Navigation timeout after {timeout} seconds"
        
        result = get_result_future.result().result
        status = get_result_future.result().status
        
        if status == 4:  # SUCCEEDED
            return True, "Navigation completed successfully"
        else:
            return False, f"Navigation failed with status {status}"
    
    def go_to_home_pose(self, skip_planning: bool = True, timeout: float = 30.0):
        """Move the robot to home pose using PlayMotion2.
        
        Args:
            skip_planning: If True, executes predefined trajectory directly without MoveIt planning.
                          If False, uses MoveIt to plan paths between waypoints.
            timeout: Maximum time to wait for motion to complete in seconds.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Wait for action server to be available
        if not self.play_motion_client.wait_for_server(timeout_sec=5.0):
            return False, "PlayMotion2 action server not available"
        
        # Create goal message
        goal_msg = PlayMotion2.Goal()
        goal_msg.motion_name = "home"
        goal_msg.skip_planning = skip_planning
        
        # Send goal
        planning_mode = "without planning" if skip_planning else "with planning"
        self.get_logger().info(f"Sending home pose goal ({planning_mode})...")
        send_goal_future = self.play_motion_client.send_goal_async(goal_msg)
        
        # Wait for goal to be accepted
        start_time = time.time()
        while not send_goal_future.done() and (time.time() - start_time) < 5.0:
            time.sleep(0.1)
        
        if not send_goal_future.done():
            return False, "Failed to send home pose goal (timeout)"
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            return False, "Home pose goal was rejected"
        
        # Wait for result
        self.get_logger().info("Home pose goal accepted, waiting for completion...")
        get_result_future = goal_handle.get_result_async()
        
        start_time = time.time()
        while not get_result_future.done() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not get_result_future.done():
            # Cancel the goal if timeout
            cancel_future = goal_handle.cancel_goal_async()
            time.sleep(0.5)  # Give it time to cancel
            return False, f"Home pose motion timeout after {timeout} seconds"
        
        result = get_result_future.result().result
        error = result.error
        
        if error == "":
            return True, f"Home pose motion completed successfully ({planning_mode})"
        else:
            return False, f"Home pose motion failed: {error}"

# Initialize ROS 2 and the Node
rclpy.init()
holland_node = HollandMcpNode()

# Run ROS 2 spin in a background thread
threading.Thread(target=lambda: rclpy.spin(holland_node), daemon=True).start()

@mcp.tool()
def move_holland(linear_speed: float = 0.2, angular_speed: float = 0.0, duration: float = 1.0) -> str:
    """
    Controls the Holland robot's movement.
    - linear_speed: Forward (+) or Backward (-) in m/s.
    - angular_speed: Left (+) or Right (-) rotation in rad/s.
    - duration: How long to move in seconds.
    """
    holland_node.execute_move(linear_speed, angular_speed, duration)
    return f"Holland Robot executed move: Linear={linear_speed}, Angular={angular_speed} for {duration}s."

@mcp.tool()
def get_holland_status() -> str:
    """Returns the current real-time coordinates of the Holland robot from AMCL."""
    # Actively fetch the latest AMCL pose
    holland_node.get_latest_amcl_pose()
    
    pos = holland_node.current_pos
    yaw_deg = math.degrees(pos['yaw']) if 'yaw' in pos and pos['yaw'] != 0.0 else 0.0
    return f"Holland Robot Position: X={pos['x']}, Y={pos['y']}, Yaw={pos.get('yaw', 0.0):.3f} rad ({yaw_deg:.1f} deg)"

@mcp.tool()
def navigate_holland(x: float, y: float, z: float = 0.0, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0, timeout: float = 60.0) -> str:
    """
    Navigate the Holland robot to a specified coordinate and orientation.
    - x: Target X coordinate in meters
    - y: Target Y coordinate in meters
    - z: Target Z coordinate in meters (usually 0.0)
    - roll: Target roll angle in radians (usually 0.0)
    - pitch: Target pitch angle in radians (usually 0.0)
    - yaw: Target yaw angle in radians (rotation around Z-axis)
    - timeout: Maximum time to wait for navigation to complete in seconds (default 60.0)
    """
    success, message = holland_node.navigate_to_pose(x, y, z, roll, pitch, yaw, timeout)
    if success:
        return f"Navigation successful: {message}"
    else:
        return f"Navigation failed: {message}"

@mcp.tool()
def go_to_home_pose(skip_planning: bool = True, timeout: float = 30.0) -> str:
    """
    Move the robot to the home pose (tucked arm position).
    - skip_planning: If True, executes predefined trajectory directly without MoveIt planning (faster, deterministic).
                    If False, uses MoveIt to plan paths between waypoints (slower, collision-aware, adaptive).
    - timeout: Maximum time to wait for motion to complete in seconds (default 30.0)
    """
    success, message = holland_node.go_to_home_pose(skip_planning, timeout)
    if success:
        return f"Home pose motion successful: {message}"
    else:
        return f"Home pose motion failed: {message}"

if __name__ == "__main__":
    mcp.run()