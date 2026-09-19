from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from neural_vector_field.config import DEFAULT_AGV_URDF, DEFAULT_RVIZ_CONFIG


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    launch_rviz = LaunchConfiguration('launch_rviz')
    launch_ps4_joy_node = LaunchConfiguration('launch_ps4_joy_node')
    launch_plot = LaunchConfiguration('launch_plot')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    agv_urdf_file = LaunchConfiguration('agv_urdf_file')

    with open(DEFAULT_AGV_URDF, 'r') as urdf_file:
        robot_description_content = urdf_file.read()

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        DeclareLaunchArgument('launch_ps4_joy_node', default_value='false'),
        DeclareLaunchArgument('launch_plot', default_value='true'),
        DeclareLaunchArgument('rviz_config_file', default_value=DEFAULT_RVIZ_CONFIG),
        DeclareLaunchArgument('agv_urdf_file', default_value=DEFAULT_AGV_URDF),

        Node(
            package='neural_vector_field',
            executable='train_vector_field_policy_entry.sh',
            name='train_vector_field_policy',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            package='neural_vector_field',
            executable='plot.sh',
            name='plot',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(launch_plot),
        ),
        Node(
            package='neural_vector_field',
            executable='plot_action.sh',
            name='plot_action',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(launch_plot),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description_content,
                'use_sim_time': use_sim_time,
            }],
        ),
        Node(
            package='agvisaac',
            executable='repubjoint',
            name='repubjoint',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            package='agvisaac',
            executable='rvizvisual',
            name='rvizvisual',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(launch_rviz),
        ),
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='log',
            condition=IfCondition(launch_ps4_joy_node),
        ),
        Node(
            package='agvisaac',
            executable='agvteleop',
            name='agvteleop',
            output='log',
            condition=IfCondition(launch_ps4_joy_node),
        ),
    ])
