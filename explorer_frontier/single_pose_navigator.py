
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Int32

import numpy as np

class Navigator(Node):

    def __init__(self):
        super().__init__('navigator_node')

        self.poses_sub = self.create_subscription(
            PoseArray,
            '/frontier_region_list',
            self.poses_callback,
            10
        )
        self.robot_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose',
            self.robot_pose_callback,
            10
        )
        self.poses_remaining = self.create_publisher(
            Int32,
            '/poses_remaining',
            10
        )
        self.robot_pose = None
        self.goal_poses = []  
        self.current_index = 0
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info("Navigator node ready, waiting for frontier poses...")

    def robot_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_pose = msg.pose.pose.position.x, msg.pose.pose.position.y

    def poses_callback(self, msg: PoseArray):
        if self.robot_pose is None:
            self.get_logger().warn("Robot pose not received yet. Setting it to zero")
            self.robot_pose = (0.0, 0.0)

        if not msg.poses:
            self.get_logger().warn("Received empty PoseArray.")
            return

        rx, ry = self.robot_pose
        distances = np.array([
            np.linalg.norm([pose.position.x - rx, pose.position.y - ry])
            for pose in msg.poses
        ])
        sorted_indices = np.argsort(distances)
        self.goal_poses = [msg.poses[i] for i in sorted_indices]
        self.current_index = 0

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Action server not available after waiting")
            return

        self.navigate_to_next()

    def navigate_to_next(self):
        if self.current_index >= len(self.goal_poses):
            self.get_logger().info("Finished navigating to all poses.")
            return

        pose = self.goal_poses[self.current_index]
        stamped = PoseStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = "map"
        stamped.pose = pose

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = stamped  # Nota: NavigateToPose usa `pose`, no `poses`

        self.get_logger().info(f"Sending pose {self.current_index+1}/{len(self.goal_poses)} to NavigateToPose...")
        self.nav_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)\
            .add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Goal {self.current_index+1} was rejected.")
            return
        self.get_logger().info("Goal accepted, waiting for result...")
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        # NavigateToPose feedback solo tiene `current_pose`, no poses restantes
        pass

    def result_callback(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info(f"Goal {self.current_index+1} completed successfully.")
        else:
            self.get_logger().warn(f"Goal {self.current_index+1} failed with code: {result.error_code}")
        
        self.current_index += 1
        self.navigate_to_next()
