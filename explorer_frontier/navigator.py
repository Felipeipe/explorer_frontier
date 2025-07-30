#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.task import Future
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from action_msgs.msg import GoalStatus
from std_msgs.msg import Int8
from nav2_msgs.action import NavigateToPose

class Navigator(Node):

    def __init__(self):
        super().__init__('navigator_node')

        # === SUBS ===
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/frontier/goal',
            self.goal_callback,
            10
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose',
            self.robot_pose_callback,
            10
        )
        # === ACTION CLIENT ===
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.robot_pose = None
        self.get_logger().info("Navigator node ready, waiting for goals...")

    # === CALLBACKS ===

    def robot_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_pose = msg.pose.pose.position

    def goal_callback(self, msg: PoseStamped):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("NavigateToPose action server not available.")
            return

        self.send_navigation_goal(msg)

    # === GOAL HANDLING ===

    def send_navigation_goal(self, pose_stamped: PoseStamped):
        goal = NavigateToPose.Goal()
        goal.pose = pose_stamped

        send_future = self.nav_client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal was rejected.")
            return

        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        pass

    def result_callback(self, future: Future):
        result_future = future.result()
        status = result_future.status
        msg = Int8()
        if status == GoalStatus.STATUS_ABORTED or status == GoalStatus.STATUS_CANCELED or GoalStatus.STATUS_UNKNOWN:
            self.get_logger().warn(f"Navigation failed with status: {status}")
            msg.data = 0
        else:
            msg.data = 1

def main(args=None):
    rclpy.init(args=args)
    navigator = Navigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()