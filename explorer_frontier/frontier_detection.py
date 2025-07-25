#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid 
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped, Pose
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
from sklearn.cluster import DBSCAN

class FastFrontPropagation(Node):

    def __init__(self):
        super().__init__('frontier_extractor_node')

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
        self.declare_parameter('unknown_cost', 0)
        self.declare_parameter('critical_cost', 50)
        self.declare_parameter('k', 2.5)
        self.declare_parameter('eps', 0.5)
        self.declare_parameter('min_samples', 1)
        self.declare_parameter('max_seeds', 1)

        self.unknown_cost = self.get_parameter('unknown_cost').value
        self.critical_cost = self.get_parameter('critical_cost').value
        self.k = self.get_parameter('k').value
        self.eps = self.get_parameter('eps').value
        self.min_samples = self.get_parameter('min_samples').value
        self.max_seeds = self.get_parameter('max_seeds').value

        self.get_logger().info(
            f'\n{self.unknown_cost = } \n{self.critical_cost = } \n{self.k = } \n{self.eps = } \n{self.min_samples = } \n{self.max_seeds = }'
        )

        self.timer = self.create_timer(3.0, self.periodic_check)
        self.marker_array = MarkerArray()
        self.pose = None
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

    # ======================== CALLBACKS ========================
    def map_callback(self, msg:OccupancyGrid):

        self.slam_map = msg.data
        self.slam_width = msg.info.width
        self.slam_height = msg.info.height
        self.slam_resolution = msg.info.resolution
        self.map_info = msg.info
        self.extract_frontier_region()

    def pose_callback(self, msg:PoseWithCovarianceStamped):
        self.pose = msg.pose.pose 

    def costmap_callback(self, msg: OccupancyGrid):
        self.costmap = msg.data
        self.cm_width = msg.info.width
        self.cm_height = msg.info.height
        self.cm_resolution = msg.info.resolution
        self.cm_info = msg.info
    
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

    def periodic_check(self):
        if len(self.F) <= 1:
            self.get_logger().info("Low number of frontiers, forcing recomputation")
            self.extract_frontier_region()

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
        if self.pose is None:
            x, y = 0.0, 0.0
        else:
            x = self.pose.position.x
            y = self.pose.position.y

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

            pose = Pose()
            pose.position.x = centroid[0]
            pose.position.y = centroid[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0  # sin orientación específica por ahora

            goal_poses.poses.append(pose)

        return goal_poses 
    # ======================== ALGORITHMS ========================
    def extract_frontier_region(self):
        self.scan_list = set() 
        self.front_queue = []
        self.F = []
        self.lattice_vector = np.full(self.slam_width*self.slam_height,-1, dtype=int)
        self.marker_array.markers = []
        # in this case, -1 represents far, 0 represents trial and 
        # 1 represents known
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
        goal_arr = self.cluster_frontiers(eps=self.eps, min_samples=self.min_samples)
        if goal_arr:
            self.goles_pub.publish(goal_arr)
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
