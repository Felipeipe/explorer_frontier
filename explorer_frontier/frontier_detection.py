#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid 
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped, Pose
import numpy as np

class FastFrontPropagation(Node):

    def __init__(self):
        super().__init__('frontier_extractor_node')
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
        self.goles_pub = self.create_publisher(
            PoseArray,
            '/frontier_region_list',
            10
        )
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            'global_costmap/obstacle_layer',
            self.costmap_callback,
            10
            )
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
    def world_to_map(self, world_x, world_y):
        mx = int((world_x - self.map_info.origin.position.x) / self.slam_resolution)
        my = int((world_y - self.map_info.origin.position.y) / self.slam_resolution)
        return mx, my
    def map_to_world(self, q):
        my, mx = self.rowcol(q)
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        wx = mx * self.slam_resolution + origin_x
        wy = my * self.slam_resolution + origin_y

        return wx, wy

    def get_seed_idx(self):
        if self.pose is None:
            x = 0.0
            y = 0.0
        else:  
            x = self.pose.position.x
            y = self.pose.position.y

        mx, my = self.world_to_map(x, y)

        max_radius = 20
        for r in range(max_radius):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = mx + dx, my + dy
                    if 0 <= nx < self.slam_width and 0 <= ny < self.slam_height:
                        idx = self.addr(nx, ny)
                        if self.slam_map[idx] == -1:
                            return idx
        return None

    def addr(self, x, y):
        return int(y*self.slam_width + x)
    
    def rowcol(self, q):
        col = q % self.slam_width
        row = q // self.slam_width
        return row, col

    def get_neighbors(self, q):
        neighbors = []
        row, col = self.rowcol(q)
        for i in range(row - 1, row + 2):      
            for j in range(col - 1, col + 2): 
                if 0 <= i < self.slam_height and 0 <= j < self.slam_width:
                    if (i, j) != (row, col):
                        neighbors.append(self.addr(j, i))
        return neighbors



    # ======================== ALGORITHMS ========================
    def extract_frontier_region(self):
        self.scan_list = set() 
        self.front_queue = []
        self.F = []
        self.lattice_vector = np.full(self.slam_width*self.slam_height,-1, dtype=int)
        # in this case, -1 represents far, 0 represents trial and 
        # 1 represents known
        self.march_front()
        self.extract_frontiers()
    def march_front(self):
        self.front_queue.append(self.get_seed_idx())
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
            neighbors = self.get_neighbors(p)
            if all(self.slam_map[idx] != 100 for idx in neighbors):
                front = Pose()
                front.position.x, front.position.y = self.map_to_world(p)
                front.position.z = 0.0
                front.orientation.x = 0.0
                front.orientation.y = 0.0
                front.orientation.z = 0.0
                front.orientation.w = 1.0
                self.F.append(front)
        pose_arr = PoseArray()
        pose_arr.poses = self.F
        pose_arr.header.stamp = self.get_clock().now().to_msg()
        pose_arr.header.frame_id = 'map'
        self.goles_pub.publish(pose_arr)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = FastFrontPropagation()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
