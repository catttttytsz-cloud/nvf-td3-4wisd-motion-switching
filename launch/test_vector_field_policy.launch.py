from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from neural_vector_field.config import (
    DEFAULT_AGV_URDF,
    DEFAULT_DATA_DIR,
    DEFAULT_RVIZ_CONFIG,
    ENABLE_TEST_DIAGNOSTIC_CSV,
    STEP_INTERVAL,
    TEST_DIAGNOSTIC_CSV_FLUSH_EVERY_N_ROWS,
)


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    launch_rviz = LaunchConfiguration('launch_rviz')
    launch_ps4_joy_node = LaunchConfiguration('launch_ps4_joy_node')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    model_path = LaunchConfiguration('model_path')
    device = LaunchConfiguration('device')
    step_dt = LaunchConfiguration('step_dt')
    enable_diagnostic_csv = LaunchConfiguration('enable_diagnostic_csv')
    diagnostic_csv_path = LaunchConfiguration('diagnostic_csv_path')
    csv_flush_every_n_rows = LaunchConfiguration('csv_flush_every_n_rows')


    with open(DEFAULT_AGV_URDF, 'r') as urdf_file:
        robot_description_content = urdf_file.read()

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        DeclareLaunchArgument('launch_ps4_joy_node', default_value='false'),
        DeclareLaunchArgument('rviz_config_file', default_value=DEFAULT_RVIZ_CONFIG),
        DeclareLaunchArgument('model_path', default_value=f'{DEFAULT_DATA_DIR}/32/epi_26100/actor.pt'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('step_dt', default_value=str(STEP_INTERVAL)),
        DeclareLaunchArgument(
            'enable_diagnostic_csv',
            default_value=str(bool(ENABLE_TEST_DIAGNOSTIC_CSV)).lower(),
        ),
        DeclareLaunchArgument(
            'diagnostic_csv_path',
            default_value=f'{DEFAULT_DATA_DIR}/test_diagnostics/step_response_diagnostics.csv',
        ),
        DeclareLaunchArgument('csv_flush_every_n_rows', default_value=str(TEST_DIAGNOSTIC_CSV_FLUSH_EVERY_N_ROWS)),

        Node(
            package='neural_vector_field',
            executable='test_vector_field_policy_entry.sh',
            name='test_vector_field_policy',
            output='screen',
            parameters=[{
                'model_path': model_path,
                'device': device,
                'step_dt': step_dt,
                'enable_diagnostic_csv': enable_diagnostic_csv,
                'diagnostic_csv_path': diagnostic_csv_path,
                'csv_flush_every_n_rows': csv_flush_every_n_rows,
            }],
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
        ),
        Node(
            package='agvisaac',
            executable='rvizvisual',
            name='rvizvisual',
            output='screen',
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
