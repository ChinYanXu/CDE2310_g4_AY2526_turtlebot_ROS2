from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction


def generate_launch_description():
    return LaunchDescription([

        # Step 1: Cartographer
        ExecuteProcess(
            cmd=[
                'ros2', 'launch',
                'turtlebot3_cartographer', 'cartographer.launch.py',
                'use_sim_time:=False'
            ],
            output='screen'
        ),

        # Step 2: Navigation2
        TimerAction(
            period=0.5,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'launch',
                        'turtlebot3_navigation2', 'navigation2.launch.py',
                        'use_sim_time:=False'
                    ],
                    output='screen'
                )
            ]
        ),

        # Step 3: Coordinator node
        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'run',
                        'cde2310_g4_ay2526', 'coordinator'
                    ],
                    output='screen'
                )
            ]
        ),

    ])