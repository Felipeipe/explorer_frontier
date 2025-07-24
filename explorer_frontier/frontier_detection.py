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
        self.unknown_cost = 0 # cost assigned to values that are unknown in costmap, might have to tune
        self.critical_cost = 45 # cost margin, discards all frontier candidates whose cost is greater
        # than this value
        self.k = 3
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
    

    def get_seed_idx(self):
        if self.pose is None:
            x = 0.0
            y = 0.0
        else:  
            x = self.pose.position.x
            y = self.pose.position.y

        mx, my = self.world_to_map(x, y, self.slam_resolution, self.map_info.origin)

        max_radius = int(self.k*np.sqrt(self.slam_height*self.slam_width))

        for r in range(max_radius):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = mx + dx, my + dy
                    if 0 <= nx < self.slam_width and 0 <= ny < self.slam_height:
                        idx = self.addr(nx, ny)
                        if self.slam_map[idx] == -1:
                            return idx
        return None

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

        pose_arr = PoseArray()
        pose_arr.poses = self.F
        pose_arr.header.stamp = self.get_clock().now().to_msg()
        pose_arr.header.frame_id = 'map'
        self.goles_pub.publish(pose_arr)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = FastFrontPropagation()
    minimal_subscriber.get_logger().info("Node initialized!")
    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
