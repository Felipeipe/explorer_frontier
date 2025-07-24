from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_path = PathJoinSubstitution([
        FindPackageShare("explorer_frontier"),
        "params",
        "frontier_params.yaml"
    ])
    frontier_node = Node(
        package="explorer_frontier",
        executable="frontier_detection",
        name="frontier_detection_node",
        output="screen",
        parameters=[config_path]
    )
    navigator_node = Node(
        package="explorer_frontier",
        executable="navigator",
        name="navigator_node",
        output="screen"
    )
    return LaunchDescription([
        frontier_node,
        navigator_node
    ])
