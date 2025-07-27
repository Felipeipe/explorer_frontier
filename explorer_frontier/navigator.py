import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateThroughPoses
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
        self.robot_pose = None
        self.goal_poses = None
        self.nav_client = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')
        self.get_logger().info("Navigator node ready, waiting for frontier poses...")

    def robot_pose_callback(self, msg:PoseWithCovarianceStamped):
        self.robot_pose = msg.pose.pose.position.x, msg.pose.pose.position.y

    def poses_callback(self, msg: PoseArray):
        if self.robot_pose is None:
            self.get_logger().warn("Robot pose not received yet. Setting it to zero")
            self.robot_pose = (0.0, 0.0)
            

        N = len(msg.poses)
        if N == 0:
            self.get_logger().warn("Received empty PoseArray.")
            return

        rx, ry = self.robot_pose

        distances = np.array([
            np.linalg.norm([pose.position.x - rx, pose.position.y - ry])
            for pose in self.F.poses
        ])

        sorted_indices = np.argsort(distances)

        sorted_poses = [self.F[i] for i in sorted_indices]

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Action server not available after waiting")
            return

        goal_msg = NavigateThroughPoses.Goal()
        for pose in sorted_poses:
            stamped = PoseStamped()
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.header.frame_id = "map"
            stamped.pose = pose
            goal_msg.poses.append(stamped)

        self.get_logger().info(f"Sending {len(goal_msg.poses)} poses to NavigateThroughPoses...")
        future = self.nav_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by NavigateThroughPoses server.')
            return

        self.get_logger().info('Goal accepted, waiting for result...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        pass
    def result_callback(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info("Navigation completed successfully!")
        else:
            self.get_logger().warn(f"Navigation failed with code: {result.error_code}")

def main(args=None):
    rclpy.init(args=args)

    navigator = Navigator()
    rclpy.spin(navigator)

    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
