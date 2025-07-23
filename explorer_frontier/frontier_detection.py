#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid 
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped

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
        self.goles = self.create_publisher(
            PoseArray,
            '/frontier_region_list',
            10
        )
        self.pose = None
        self.map = None
        self.lattice_vector = ...
        self.scan_list = []
        self.front_queue = []
        self.F = []
        self.width = None
        self.height = None
        self.resolution = None
    
    def map_callback(self, msg:OccupancyGrid):
        self.map = msg.data
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.get_logger().info(f"{self.width = } \n{self.height = }")
        self.extract_frontier_region()

    def pose_callback(self, msg:PoseWithCovarianceStamped):
        self.pose = msg.pose.pose 

    def addr(self, x, y):
        return y*self.width + x


    def extract_frontier_region(self):
        self.scan_list = [] 
        self.front_queue = []
        self.F = []
        self.march_front()
        self.extract_frontiers()
    def march_front(self):
        self.front_queue.append(self.get_seed_idx())
        while self.front_queue:
            q = self.front_queue.pop()

        pass
    def extract_frontiers(self):
        pass
    

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