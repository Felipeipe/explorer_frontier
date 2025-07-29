from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_path = PathJoinSubstitution([
        FindPackageShare("explorer_frontier"),
        "params",
        "frontier_detection.yaml"
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
    single_pose_navigator = Node(
        package='explorer_frontier',
        executable='single_pose_navigator',
        name='single_pose_navigator_node',
        output='screen',
        parameters=[config_path]
    )
    return LaunchDescription([
        frontier_node,
        navigator_node
    ])
