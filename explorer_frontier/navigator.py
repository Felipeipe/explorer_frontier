import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseArray
from nav2_msgs.action import NavigateThroughPoses

class Navigator(Node):

    def __init__(self):
        super().__init__('navigator_node')

        self.poses_sub = self.create_subscription(
            PoseArray,
            '/frontier_region_list',
            self.poses_callback,
            10
        )

        self.nav_client = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')
        self.get_logger().info("Navigator node ready, waiting for frontier poses...")

    def poses_callback(self, msg: PoseArray):
        if not msg.poses:
            self.get_logger().warn("Received empty PoseArray!")
            return

        self.get_logger().info("Waiting for navigate_through_poses action server...")
        self.nav_client.wait_for_server()

        goal_msg = NavigateThroughPoses.Goal()
        goal_msg.poses = msg.poses
        goal_msg.behavior_tree = ''

        self.get_logger().info(f"Sending {len(msg.poses)} poses to NavigateThroughPoses...")
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by NavigateThroughPoses server.')
            return

        self.get_logger().info('Goal accepted, waiting for result...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose.pose
        self.get_logger().info(
            f"Current pose: x={current_pose.position.x:.2f}, y={current_pose.position.y:.2f}"
        )

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
