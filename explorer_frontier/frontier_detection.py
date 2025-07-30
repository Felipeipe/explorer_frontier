#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid 
from nav2_msgs.action import NavigateThroughPoses
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped, Pose, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Int8
from tf_transformations import quaternion_from_euler
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
        self.goal_status_sub = self.create_subscription(
            Int8,
            '/frontier/goal_status',
            self.goal_status_callback,
            10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10)

        # =============== PUBLISHERS ====================
        
        self.goles_pub = self.create_publisher(
            PoseStamped,
            '/frontier/goal',
            10
        )
        self.clusters = self.create_publisher(
            PoseArray,
            '/frontier/frontiers',
            10
        )
        self.seed_idx_pub = self.create_publisher(
            MarkerArray,
            '/frontier/seed_indices',
            10
        )
        # Parameters, see params/frontier_detection.yaml
        self.declare_parameter('unknown_cost',               0)
        self.declare_parameter('critical_cost',             50)
        self.declare_parameter('k',                        2.5)
        self.declare_parameter('eps',                      0.5)
        self.declare_parameter('min_samples',                1)
        self.declare_parameter('max_seeds',                  1)
        self.declare_parameter('set_frontier_permanence', True)
        self.declare_parameter('cost_window',                3)

        self.unknown_cost            = self.get_parameter('unknown_cost').value
        self.critical_cost           = self.get_parameter('critical_cost').value
        self.k                       = self.get_parameter('k').value
        self.eps                     = self.get_parameter('eps').value
        self.min_samples             = self.get_parameter('min_samples').value
        self.max_seeds               = self.get_parameter('max_seeds').value
        self.set_frontier_permanence = self.get_parameter('set_frontier_permanence').value
        self.cost_window             = self.get_parameter('cost_window').value

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
        self.first_run = True
        self.goal_status = None

    # ======================== CALLBACKS ========================
    def pose_callback(self, msg:PoseWithCovarianceStamped):
        self.robot_pose = msg.pose.pose 

    def costmap_callback(self, msg: OccupancyGrid):
        self.costmap = msg.data
        self.cm_width = msg.info.width
        self.cm_height = msg.info.height
        self.cm_resolution = msg.info.resolution
        self.cm_info = msg.info

    def goal_status_callback(self, msg:Int8):
        self.goal_status = msg.data

    def map_callback(self, msg:OccupancyGrid):
        padded_data, new_width, new_height = self.pad_map_with_unknown(
            msg.data,
            msg.info.width,
            msg.info.height
        )

        self.slam_map = padded_data
        self.slam_width = new_width
        self.slam_height = new_height
        self.slam_resolution = msg.info.resolution

        self.map_info = msg.info
        self.map_info.width = new_width
        self.map_info.height = new_height
        self.map_info.origin.position.x -= self.slam_resolution
        self.map_info.origin.position.y -= self.slam_resolution

        self.extract_frontier_region()

    # ======================== UTILITIES ========================
    def pad_map_with_unknown(self, map_data, width, height, pad_value=-1):
        new_width = width + 2
        new_height = height + 2

        padded_map = np.full((new_height, new_width), pad_value, dtype=np.int8)

        original_map = np.array(map_data, dtype=np.int8).reshape((height, width))

        padded_map[1:height+1, 1:width+1] = original_map

        return padded_map.flatten().tolist(), new_width, new_height

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
            return float('inf')
            
        wx, wy = self.map_to_world(q, self.slam_resolution, self.map_info.origin)
        mcx, mcy = self.world_to_map(wx, wy, self.cm_resolution, self.cm_info.origin) 
        qcm = self.addr(mcx, mcy, self.cm_width)
        neig = self.get_neighbors(qcm, self.cm_width, self.cm_height, self.cost_window)
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
        # seeds = [0]
        return seeds


    def nth_nearest(self, poses: list[PoseStamped], n = 0):

        if self.robot_pose is None:
            rx, ry = (0.0, 0.0)
        else:
            rx, ry = self.robot_pose.position.x, self.robot_pose.position.y

        if not poses:
            self.get_logger().warn("Received empty PoseArray.")
            return

        distances = np.array([
            np.linalg.norm([pose.pose.position.x - rx, pose.pose.position.y - ry])
            for pose in poses
        ])
        sorted_indices = np.argsort(distances)
        return poses[sorted_indices[n]]

    


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

    def get_neighbors(self, q, width=None, height=None, radius=1):
        if width is None:
            width = self.slam_width
        if height is None:
            height = self.slam_height

        neighbors = []
        row, col = self.rowcol(q, width)
        for i in range(row - radius, row + radius + 1):      
            for j in range(col - radius, col + radius + 1): 
                if 0 <= i < height and 0 <= j < width:
                    if (i, j) != (row, col):
                        neighbors.append(self.addr(j, i, width))
        return neighbors

    def cluster_frontiers(self, eps=0.5, min_samples=3) -> list[PoseStamped]:
        if not self.F:
            return []
        points = np.array([[p.position.x, p.position.y] for p in self.F])

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        unique_labels = set(labels) - {-1}

        goal_poses = []

        for label in unique_labels:
            cluster_points = points[labels == label]
            centroid = np.median(cluster_points, axis=0)

            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = centroid[0]
            pose.pose.position.y = centroid[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  
            goal_poses.append(pose)
        return goal_poses 

    # ======================== ALGORITHMS ========================
    def extract_frontier_region(self):

        prev_frontiers = self.F.copy()
        valid_frontiers = []

        if self.set_frontier_permanence:
            for pose in prev_frontiers:
                mx, my = self.world_to_map(pose.position.x, pose.position.y, self.slam_resolution, self.map_info.origin)
                if 0 <= mx < self.slam_width and 0 <= my < self.slam_height:
                    idx = self.addr(mx, my)
                    if self.slam_map[idx] == -1:
                        valid_frontiers.append(pose)

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
            neighbors = self.get_neighbors(p, None, None, radius=2)
            cost = self.get_cost(p)
            if cost < self.critical_cost: 
                if all(self.slam_map[idx] < 90 for idx in neighbors):
                    front = Pose()
                    front.position.x, front.position.y = self.map_to_world(p, self.slam_resolution, self.map_info.origin)
                    front.position.z = 0.0
                    front.orientation.x = 0.0
                    front.orientation.y = 0.0
                    front.orientation.z = 0.0
                    front.orientation.w = 1.0
                    self.F.append(front) 

        clusters = self.cluster_frontiers(eps=self.eps, min_samples=self.min_samples)
        if not clusters:
            self.get_logger().warn("No frontier clusters found.")
            return
        i = 0
        if self.goal_status == 0 and i < len(clusters) - 1:
            i += 1
        self.goal_pose = self.nth_nearest(clusters, i)

        self.goles_pub.publish(self.goal_pose)
        pose_arr = PoseArray()
        pose_arr.header.stamp = self.get_clock().now().to_msg()
        pose_arr.header.frame_id = 'map'
        pose_arr.poses = self.F
        self.clusters.publish(pose_arr)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = FastFrontPropagation()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
