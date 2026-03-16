#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    project_pkg = get_package_share_directory('cde2310_g4_AY2526')
    tb3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    tb3_desc_pkg = get_package_share_directory('turtlebot3_description')
    ros_gz_sim_pkg = get_package_share_directory('ros_gz_sim')

    launch_file_dir = os.path.join(tb3_gazebo_pkg, 'launch')

    world = os.path.join(
        project_pkg,
        'worlds',
        'custom_maze_CDE2310.sdf'
    )

    rviz_config = os.path.join(
        project_pkg,
        'rviz',
        'tb3_cartographer.rviz'
    )

    cartographer_config_dir = os.path.join(project_pkg, 'config')
    cartographer_config_basename = 'turtlebot3_lds_2d.lua'

    tb3_gazebo_parent = os.path.dirname(tb3_gazebo_pkg)
    tb3_desc_parent = os.path.dirname(tb3_desc_pkg)

    set_gz_resource = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=':'.join([
            tb3_gazebo_parent,
            tb3_desc_parent,
        ])
    )

    set_ign_resource = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=':'.join([
            tb3_gazebo_parent,
            tb3_desc_parent,
        ])
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r -v4 {world}',
            'on_exit_shutdown': 'true'
        }.items()
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true'
        }.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': '0.0',
            'y_pose': '0.0'
        }.items()
    )

    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '-configuration_directory', cartographer_config_dir,
            '-configuration_basename', cartographer_config_basename,
        ],
        remappings=[
            ('scan', '/scan'),
            ('odom', '/odom'),
        ]
    )

    cartographer_occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '-resolution', '0.05',
            '-publish_period_sec', '1.0'
        ]
    )

    rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        set_gz_resource,
        set_ign_resource,
        gazebo,
        robot_state_publisher_cmd,
        spawn_turtlebot_cmd,
        cartographer_node,
        cartographer_occupancy_grid_node,
        rviz_cmd,
    ])