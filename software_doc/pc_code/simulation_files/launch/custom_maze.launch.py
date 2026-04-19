#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():
    proj_pkg = get_package_share_directory('cde2310_g4_ay2526')
    tb3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    tb3_desc_pkg = get_package_share_directory('turtlebot3_description')
    ros_gz_sim_pkg = get_package_share_directory('ros_gz_sim')
    slam_toolbox_pkg = get_package_share_directory('slam_toolbox')
    nav2_pkg = get_package_share_directory('turtlebot3_navigation2')

    launch_file_dir = os.path.join(tb3_gazebo_pkg, 'launch')

    world = os.path.join(
        proj_pkg,
        'worlds',
        'custom_maze_CDE2310.sdf'
    )

    # Resource paths
    tb3_gazebo_parent = os.path.dirname(tb3_gazebo_pkg)
    tb3_desc_parent = os.path.dirname(tb3_desc_pkg)

    set_gz_resource = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=':'.join([tb3_gazebo_parent, tb3_desc_parent])
    )

    set_ign_resource = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=':'.join([tb3_gazebo_parent, tb3_desc_parent])
    )

    # Gazebo
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v4 ', world], 'on_exit_shutdown': 'true'}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v4 '}.items()
    )

    # Robot
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
        launch_arguments={'x_pose': '0.0', 'y_pose': '0.0'}.items()
    )

    # SLAM Toolbox
    slam_toolbox_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_pkg, 'launch', 'online_async_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'True'
        }.items()
    )

    # Nav2
    nav2_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_pkg, 'launch', 'navigation2.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'True',
            'slam': 'True'
        }.items()
    )

    # Frontier Explorer Node
    frontier_explorer_node = Node(
        package='cde2310_g4_ay2526',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen'
    )

    return LaunchDescription([
        set_gz_resource,
        set_ign_resource,

        gzserver_cmd,
        gzclient_cmd,

        robot_state_publisher_cmd,
        spawn_turtlebot_cmd,

        # Delay SLAM (wait for robot + topics to exist)
        TimerAction(
            period=10.0,
            actions=[slam_toolbox_cmd]
        ),

        # Delay Nav2 (wait for SLAM to start publishing map + TF)
        TimerAction(
            period=11.0,
            actions=[nav2_cmd]
        ),

        # Delay Frontier Explorer (wait for Nav2 to be ready)
        TimerAction(
            period=15.0,
            actions=[frontier_explorer_node]
        ),
    ])