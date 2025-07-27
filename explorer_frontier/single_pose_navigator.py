
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose, Spin
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
        self.declare_parameter('similarity_tolerance', 0.005)
        self.declare_parameter('spin_wait_time', 5.0)
        self.tolerance = self.get_parameter('similarity_tolerance').value
        self.spin_wait_time = self.get_parameter('spin_wait_time').value

        self.robot_pose = None
        self.goal_poses = []  
        self.current_index = 0
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.navigating = False
        self.last_goal_poses = []
        self.get_logger().info("Navigator node ready, waiting for frontier poses...")

    def robot_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_pose = msg.pose.pose.position.x, msg.pose.pose.position.y

    def send_spin_goal(self, radians=6.28, time_allowance_sec=10.0):
        if not self.spin_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn("Spin action server not available.")
            return

        goal_msg = Spin.Goal()
        goal_msg.target_yaw = radians
        goal_msg.time_allowance.sec = int(time_allowance_sec)

        self.get_logger().info(f"Sending spin goal: {radians:.2f} rad")
        spin_future = self.spin_client.send_goal_async(goal_msg)
        spin_future.add_done_callback(self.spin_response_callback)

    def spin_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Spin goal was rejected.")
            return

        self.get_logger().info("Spin goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.spin_result_callback)

    def spin_result_callback(self, future):
        result = future.result().result
        self.get_logger().info("Spin completed successfully.")


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

        if self.same_poses(self.goal_poses, self.last_goal_poses):
            self.get_logger().info("Received same poses as last time. Skipping...")
            return

        self.last_goal_poses = self.goal_poses
        self.current_index = 0
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Action server not available after waiting")
            return

        self.navigate_to_next()

    def same_poses(self, poses1, poses2):
        if len(poses1) != len(poses2):
            return False
        for p1, p2 in zip(poses1, poses2):
            if abs(p1.position.x - p2.position.x) > self.tolerance or abs(p1.position.y - p2.position.y) > self.tolerance:
                return False
        return True

    def navigate_to_next(self):
        if self.current_index >= len(self.goal_poses):
            self.get_logger().info("Finished navigating to all poses.")
            return
        self.navigating = True
        pose = self.goal_poses[self.current_index]
        stamped = PoseStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = "map"
        stamped.pose = pose

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = stamped 

        self.get_logger().info(f"Sending pose {self.current_index+1}/{len(self.goal_poses)} to NavigateToPose...")
        goal_async = self.nav_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        goal_async.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Goal {self.current_index+1} was rejected.")
            return
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        self.distance_remaining = feedback_msg.feedback.distance_remaining
        if self.distance_remaining < 0.4 and self.navigating:
            self.navigating = False
            self.get_logger().info(f"Goal {self.current_index+1} completed successfully.")
            self.current_index += 1
            self.navigate_to_next()
    def result_callback(self, future):
        result = future.result().result 
        if result.error_code != 0:
            self.get_logger().warn(f"Navigation failed with error code: {result.error_code}")


def main(args=None):
    rclpy.init(args=args)

    navigator = Navigator()
    rclpy.spin(navigator)

    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()