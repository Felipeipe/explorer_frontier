#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid 
from nav2_msgs.action import NavigateThroughPoses
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped, Pose, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker

import numpy as np
from sklearn.cluster import DBSCAN


class FastFrontPropagation(Node):

    def __init__(self):
        super().__init__('frontier_extractor_node')
        # ===============   SERVICES  ==================== 
        self.nav_client = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Action server not available after waiting")
        # =============== SUBSCRIBERS ====================
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'pose',
            self.pose_callback,
            10,
        )
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            'global_costmap/costmap',
            self.costmap_callback,
            10
        )
        self.get_logger().info("After subscribers")
        # =============== PUBLISHERS ====================

        self.goles_pub = self.create_publisher(
            PoseArray,
            '/frontier_region_list',
            10
        )
        self.clusters = self.create_publisher(
            PoseArray,
            '/frontiers',
            10
        )
        self.seed_idx_pub = self.create_publisher(
            MarkerArray,
            '/seed_indices',
            10
        )
        # Parameters, see params/frontier_detection.yaml
        self.declare_parameter('unknown_cost', 0)
        self.declare_parameter('critical_cost', 50)
        self.declare_parameter('k', 2.5)
        self.declare_parameter('eps', 0.5)
        self.declare_parameter('min_samples', 1)
        self.declare_parameter('max_seeds', 1)
        self.declare_parameter('number_of_map_updates', 5)

        self.unknown_cost = self.get_parameter('unknown_cost').value
        self.critical_cost = self.get_parameter('critical_cost').value
        self.k = self.get_parameter('k').value
        self.eps = self.get_parameter('eps').value
        self.min_samples = self.get_parameter('min_samples').value
        self.max_seeds = self.get_parameter('max_seeds').value
        self.map_updates = self.get_parameter('number_of_map_updates').value

        # Auxiliary variables
        self.marker_array = MarkerArray()
        self.robot_pose = None
        self.slam_map = None
        self.lattice_vector = None 
        self.scan_list = set()
        self.front_queue = []
        self.F = []
        self.slam_width = None
        self.slam_height = None
        self.slam_resolution = None
        self.map_info = None
        self.costmap = None
        self.cm_height = None
        self.cm_width = None
        self.cm_resolution = None
        self.cm_info = None
        # 0: success, 1: waiting, -1: error
        self.nav_status = 1
        self.initialized = False

    # ======================== CALLBACKS ========================
    def map_callback(self, msg:OccupancyGrid):
        self.slam_map = msg.data
        self.slam_width = msg.info.width
        self.slam_height = msg.info.height
        self.slam_resolution = msg.info.resolution
        self.map_info = msg.info 
        if self.nav_status == 0 or not self.initialized:
            self.initialized = True
            self.extract_frontier_region()

    def pose_callback(self, msg:PoseWithCovarianceStamped):
        self.robot_pose = msg.pose.pose 

    def costmap_callback(self, msg: OccupancyGrid):
        self.costmap = msg.data
        self.cm_width = msg.info.width
        self.cm_height = msg.info.height
        self.cm_resolution = msg.info.resolution
        self.cm_info = msg.info
    
    def go_to_frontiers(self):
        if self.goal_poses is None or not self.goal_poses.poses:
            self.get_logger().warn("No goals available for navigation.")
            # self.go_to_frontiers()
        if self.robot_pose is None:
            self.get_logger().warn("Robot pose not received yet. Setting it to zero")
            self.robot_pose = (0.0, 0.0)
        rx, ry = self.robot_pose

        distances = np.array([
            np.linalg.norm([pose.pose.position.x - rx, pose.pose.position.y - ry])
            for pose in self.goal_poses.poses
        ])

        sorted_indices = np.argsort(distances)

        sorted_poses = [self.goal_poses.poses[i] for i in sorted_indices]


        goal_msg = NavigateThroughPoses.Goal()
        # for pose in sorted_poses:
        #     stamped = PoseStamped()
        #     stamped.header.stamp = self.get_clock().now().to_msg()
        #     stamped.header.frame_id = "map"
        #     stamped.pose = pose
        #     goal_msg.poses.append(stamped)
        goal_msg.poses = sorted_poses
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
        feedback = feedback_msg.feedback
        print(f"[Feedback] Navegando hacia la pose {feedback.current_pose}")

    def result_callback(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info("Navigation completed successfully!")
            self.nav_status = 0
        else:
            self.get_logger().warn(f"Navigation failed with code: {result.error_code}")

    # ======================== UTILITIES ========================
    def seed_to_marker(self, q):
        wx, wy = self.map_to_world(q, self.slam_resolution, self.map_info.origin)

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "seed_points"
        marker.id = q
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = wx
        marker.pose.position.y = wy
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        return marker

    def world_to_map(self, world_x, world_y, resolution, origin):
        mx = int((world_x - origin.position.x) / resolution)
        my = int((world_y - origin.position.y) / resolution)
        return mx, my
    def map_to_world(self, q, resolution, origin):
        my, mx = self.rowcol(q)
        origin_x = origin.position.x
        origin_y = origin.position.y

        wx = mx * resolution + origin_x
        wy = my * resolution + origin_y

        return wx, wy

    def get_cost(self, q):
        """
        Calculates average obstacle cost in a small patch @ index q
        """
        if self.cm_info is None or self.costmap is None:
            self.get_logger().warn("Costmap not received yet, skipping cost calculation.")
            return float('inf')
            
        wx, wy = self.map_to_world(q, self.slam_resolution, self.map_info.origin)
        mcx, mcy = self.world_to_map(wx, wy, self.cm_resolution, self.cm_info.origin) 
        qcm = self.addr(mcx, mcy, self.cm_width)
        neig = self.get_neighbors(qcm, self.cm_width, self.cm_height)
        cost = 0
        n = len(neig)
        for idx in neig:
            if 0 <= idx < len(self.costmap):
                if self.costmap[idx] == -1:
                    cost += self.unknown_cost / n
                else:
                    cost += self.costmap[idx] / n
            else:
                cost += self.unknown_cost / n 
        return cost

    def get_seed_indices(self, max_seeds=None):
        if max_seeds == None:
            max_seeds = self.max_seeds
        if self.robot_pose is None:
            x, y = 0.0, 0.0
        else:
            x = self.robot_pose.position.x
            y = self.robot_pose.position.y

        mx, my = self.world_to_map(x, y, self.slam_resolution, self.map_info.origin)
        max_radius = int(self.k * np.sqrt(self.slam_height * self.slam_width))

        seeds = []
        for r in range(max_radius):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = mx + dx, my + dy
                    if 0 <= nx < self.slam_width and 0 <= ny < self.slam_height:
                        idx = self.addr(nx, ny)
                        if self.slam_map[idx] == -1 and self.lattice_vector[idx] == -1:
                            seeds.append(idx)
                            self.marker_array.markers.append(self.seed_to_marker(idx))
                            if len(seeds) >= max_seeds:
                                self.seed_idx_pub.publish(self.marker_array)
                                return seeds

        self.seed_idx_pub.publish(self.marker_array)
        return seeds



    def addr(self, x, y, width = None):
        if width == None:
            width = self.slam_width
        return int(y*width + x)
    
    def rowcol(self, q, width=None):
        if width == None:
            width = self.slam_width
        col = q % width
        row = q // width
        return row, col

    def get_neighbors(self, q, width = None, height = None):
        if width == None:
            width = self.slam_width
        if height == None:
            height = self.slam_height
        neighbors = []
        row, col = self.rowcol(q, width)
        for i in range(row - 1, row + 2):      
            for j in range(col - 1, col + 2): 
                if 0 <= i < height and 0 <= j < width:
                    if (i, j) != (row, col):
                        neighbors.append(self.addr(j, i))
        return neighbors


    def cluster_frontiers(self, eps=0.5, min_samples=3):
        if not self.F:
            return None
        points = np.array([[p.position.x, p.position.y] for p in self.F])

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        unique_labels = set(labels) - {-1}

        goal_poses = PoseArray()
        goal_poses.header.stamp = self.get_clock().now().to_msg()
        goal_poses.header.frame_id = 'map'

        for label in unique_labels:
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)

            pose = PoseStamped()
            pose.pose.position.x = centroid[0]
            pose.pose.position.y = centroid[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            goal_poses.poses.append(pose)

        return goal_poses 
    # ======================== ALGORITHMS ========================
    def extract_frontier_region(self):

        prev_frontiers = self.F.copy()
        valid_frontiers = []

        # for pose in prev_frontiers:
        #     mx, my = self.world_to_map(pose.position.x, pose.position.y, self.slam_resolution, self.map_info.origin)
        #     if 0 <= mx < self.slam_width and 0 <= my < self.slam_height:
        #         idx = self.addr(mx, my)
        #         if self.slam_map[idx] == -1:
        #             valid_frontiers.append(pose)

        self.scan_list = set()
        self.front_queue = []
        self.F = valid_frontiers
        self.lattice_vector = np.full(self.slam_width * self.slam_height, -1, dtype=int)
        self.marker_array.markers = []

        self.march_front()
        self.extract_frontiers()

    def march_front(self):
        seed_indices = self.get_seed_indices()
        self.front_queue.extend(seed_indices)
        for s in seed_indices:
            self.lattice_vector[s] = 1

        while self.front_queue:
            # ma = self.list_to_marker_array(self.front_queue)
            # self.seed_idx_pub.publish(ma)
            q = self.front_queue.pop()

            self.lattice_vector[q] = 1
            neighbors = self.get_neighbors(q)
            for idx in neighbors:
                if self.lattice_vector[idx] != 1:
                    if self.slam_map[idx] == -1:
                        if self.lattice_vector[idx] == -1:
                            self.front_queue.append(idx)
                            self.lattice_vector[idx] = 0
                    else:
                        self.scan_list.add(q)

    def extract_frontiers(self):
        for p in self.scan_list:
            neighbors = self.get_neighbors(p)
            cost = self.get_cost(p)
            if cost < self.critical_cost: 
                if all(self.slam_map[idx] != 100 for idx in neighbors):
                    front = Pose()
                    front.position.x, front.position.y = self.map_to_world(p, self.slam_resolution, self.map_info.origin)
                    front.position.z = 0.0
                    front.orientation.x = 0.0
                    front.orientation.y = 0.0
                    front.orientation.z = 0.0
                    front.orientation.w = 1.0
                    self.F.append(front)
        self.goal_poses = self.cluster_frontiers(eps=self.eps, min_samples=self.min_samples)
        pose_arr = PoseArray()
        pose_arr.header.stamp = self.get_clock().now().to_msg()
        pose_arr.header.frame_id = 'map'
        pose_arr.poses = self.F
        self.clusters.publish(pose_arr)
        self.go_to_frontiers()

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = FastFrontPropagation()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
